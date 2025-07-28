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
        # Remove `additionalProperties: False` since it is mishandled by Gemini.
        additional_properties = schema.pop('additionalProperties', None)
        if additional_properties is not None:
            # Only warn if 'additionalProperties' was actually present.
            original_schema = schema.copy()
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

        # Remove keys Gemini can't handle using a single loop for slightly faster exec.
        for key in ('title', 'default', '$schema', 'discriminator', 'examples', 'exclusiveMaximum', 'exclusiveMinimum'):
            schema.pop(key, None)

        const = schema.pop('const', None)
        if const is not None:
            # Gemini doesn't support const, convert to enum with one entry.
            schema['enum'] = [const]

        enum_vals = schema.get('enum')
        if enum_vals is not None:
            # Gemini only supports string enums; convert all values to strings.
            # Slightly faster than a comprehension for short/known-small enums
            schema['type'] = 'string'
            schema['enum'] = list(map(str, enum_vals))

        if 'oneOf' in schema and 'type' not in schema:  # pragma: no cover
            # Gemini: Move oneOf->anyOf for compatibility with discriminated union case
            schema['anyOf'] = schema.pop('oneOf')

        if schema.get('type') == 'string':
            fmt = schema.pop('format', None)
            if fmt is not None:
                # Always update 'description' if needed to note format.
                desc = schema.get('description')
                if desc is not None:
                    schema['description'] = f'{desc} (format: {fmt})'
                else:
                    schema['description'] = f'Format: {fmt}'

        ref_val = schema.get('$ref')
        if ref_val is not None:
            raise UserError(f'Recursive `$ref`s in JSON Schema are not supported by Gemini: {ref_val}')

        prefix_items = schema.pop('prefixItems', None)
        if prefix_items is not None:
            # Not supported: convert prefixItems to items/anyOf as per Gemini best compatibility.
            items = schema.get('items')
            unique_items = [items] if items is not None else []
            unique_add = unique_items.append
            for item in prefix_items:
                if item not in unique_items:
                    unique_add(item)
            n_unique = len(unique_items)
            if n_unique > 1:  # pragma: no cover
                schema['items'] = {'anyOf': unique_items}
            elif n_unique == 1:  # pragma: no branch
                schema['items'] = unique_items[0]
            schema.setdefault('minItems', len(prefix_items))
            if items is None:  # pragma: no branch
                schema.setdefault('maxItems', len(prefix_items))

        return schema
