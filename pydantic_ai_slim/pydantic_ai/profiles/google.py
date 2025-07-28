from __future__ import annotations as _annotations

import warnings

from pydantic_ai.exceptions import UserError

from . import ModelProfile
from ._json_schema import JsonSchema, JsonSchemaTransformer


def google_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Google model."""
    return ModelProfile(
        json_schema_transformer=GoogleJsonSchemaTransformer,
        supports_json_schema_output=True,
        supports_json_object_output=True,
    )


class GoogleJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms the JSON Schema from Pydantic to be suitable for Gemini.

    Gemini which [supports](https://ai.google.dev/gemini-api/docs/function-calling#function_declarations)
    a subset of OpenAPI v3.0.3.

    Specifically:
    * gemini doesn't allow the `title` keyword to be set
    * gemini doesn't allow `$defs` — we need to inline the definitions where possible
    """

    def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
        super().__init__(schema, strict=strict, prefer_inlined_defs=True, simplify_nullable_unions=True)

    def transform(self, schema: JsonSchema) -> JsonSchema:
        # Remove 'additionalProperties' and warn if present
        if 'additionalProperties' in schema:
            additional_properties = schema['additionalProperties']
            if additional_properties:
                original_schema = dict(schema)
                original_schema['additionalProperties'] = additional_properties
                warnings.warn(
                    '`additionalProperties` is not supported by Gemini; it will be removed from the tool JSON schema.'
                    f' Full schema: {self.schema}\n\n'
                    f'Source of additionalProperties within the full schema: {original_schema}\n\n'
                    'If this came from a field with a type like `dict[str, MyType]`, that field will always be empty.\n\n'
                    "If Google's APIs are updated to support this properly, please create an issue on the Pydantic AI GitHub"
                    ' and we will fix this behavior.',
                    UserWarning,
                )
            schema.pop('additionalProperties', None)

        # Bulk pop keys known to simply be removed if present
        for key in ('title', 'default', '$schema', 'discriminator', 'examples', 'exclusiveMaximum', 'exclusiveMinimum'):
            schema.pop(key, None)

        # Convert 'const' to single-value 'enum'
        const_val = schema.pop('const', None)
        if const_val is not None:
            schema['enum'] = [const_val]

        # Enforce Gemini string-enum constraint
        # Use fast-path in-place string conversion
        enum = schema.get('enum')
        if enum is not None:
            if schema.get('type') != 'string' or any(not isinstance(v, str) for v in enum):
                schema['type'] = 'string'
                schema['enum'] = [str(val) for val in enum]

        # oneOf to anyOf if no top-level 'type'
        if 'oneOf' in schema and 'type' not in schema:
            schema['anyOf'] = schema.pop('oneOf')

        # Move format to description for strings
        if schema.get('type') == 'string' and 'format' in schema:
            fmt = schema.pop('format')
            desc = schema.get('description')
            if desc:
                schema['description'] = f'{desc} (format: {fmt})'
            else:
                schema['description'] = f'Format: {fmt}'

        # Check for recursive refs (Gemini does not support)
        if '$ref' in schema:
            raise UserError(f'Recursive `$ref`s in JSON Schema are not supported by Gemini: {schema["$ref"]}')

        # PrefixItems: collapse to items/anyOf
        if 'prefixItems' in schema:
            prefix_items = schema.pop('prefixItems')
            items = schema.get('items')
            if items is not None:
                # Avoid duplicates only if required, usually a rare case
                unique_items = [items] + [item for item in prefix_items if item != items]
            else:
                unique_items = list(prefix_items)
            n = len(unique_items)
            if n > 1:
                schema['items'] = {'anyOf': unique_items}
            elif n == 1:
                schema['items'] = unique_items[0]
            schema.setdefault('minItems', len(prefix_items))
            if items is None:
                schema.setdefault('maxItems', len(prefix_items))

        return schema
