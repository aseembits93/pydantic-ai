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
        # Remove artificial delay for runtime optimization

        # Remove unsupported 'additionalProperties' with warning if present
        additional_properties = schema.pop('additionalProperties', None)
        if additional_properties is not None:
            original_schema = {**schema, 'additionalProperties': additional_properties}
            warnings.warn(
                '`additionalProperties` is not supported by Gemini; it will be removed from the tool JSON schema.'
                f' Full schema: {self.schema}\n\n'
                f'Source of additionalProperties within the full schema: {original_schema}\n\n'
                'If this came from a field with a type like `dict[str, MyType]`, that field will always be empty.\n\n'
                "If Google's APIs are updated to support this properly, please create an issue on the Pydantic AI GitHub"
                ' and we will fix this behavior.',
                UserWarning,
            )

        # Batch-remove unsupported fields for speed (avoiding redundant hash lookups)
        for key in ('title', 'default', '$schema', 'discriminator', 'examples', 'exclusiveMaximum', 'exclusiveMinimum'):
            schema.pop(key, None)

        # Gemini doesn't support const, but supports enum with a single value
        const = schema.pop('const', None)
        if const is not None:
            schema['enum'] = [const]

        # Gemini only supports string enums, convert any to string if present
        enum = schema.get('enum')
        if enum:
            schema['type'] = 'string'
            # skip conversion if already all strings for efficiency
            if any(not isinstance(val, str) for val in enum):
                schema['enum'] = [str(val) for val in enum]

        # Discriminated union fix: oneOf->anyOf when type is missing
        if 'oneOf' in schema and 'type' not in schema:  # pragma: no cover
            schema['anyOf'] = schema.pop('oneOf')

        # Format: append format to description for string types
        if schema.get('type') == 'string':
            fmt = schema.pop('format', None)
            if fmt:
                description = schema.get('description')
                if description:
                    schema['description'] = f'{description} (format: {fmt})'
                else:
                    schema['description'] = f'Format: {fmt}'

        # Gemini does not support recursive $ref
        ref = schema.get('$ref')
        if ref is not None:
            raise UserError(f'Recursive `$ref`s in JSON Schema are not supported by Gemini: {ref}')

        # prefixItems -> items conversion for Gemini compatibility
        prefix_items = schema.pop('prefixItems', None)
        if prefix_items is not None:
            items = schema.get('items')
            unique_items = []
            if items is not None:
                unique_items.append(items)
            for item in prefix_items:
                if item not in unique_items:
                    unique_items.append(item)
            if len(unique_items) > 1:  # pragma: no cover
                schema['items'] = {'anyOf': unique_items}
            elif len(unique_items) == 1:  # pragma: no branch
                schema['items'] = unique_items[0]
            schema.setdefault('minItems', len(prefix_items))
            if items is None:  # pragma: no branch
                schema.setdefault('maxItems', len(prefix_items))

        return schema
