from __future__ import annotations as _annotations

import inspect
import json
from collections.abc import Awaitable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, Callable, Generic, Literal, Union, cast

from pydantic import TypeAdapter, ValidationError
from pydantic_core import SchemaValidator
from typing_extensions import TypeAliasType, TypedDict, TypeVar, get_args, get_origin
from typing_inspection import typing_objects
from typing_inspection.introspection import is_union_origin

from pydantic_ai.profiles import ModelProfile

from . import _function_schema, _utils, messages as _messages
from .exceptions import ModelRetry, UserError
from .tools import AgentDepsT, GenerateToolJsonSchema, ObjectJsonSchema, RunContext, ToolDefinition

T = TypeVar('T')
"""An invariant TypeVar."""
OutputDataT_inv = TypeVar('OutputDataT_inv', default=str)
"""
An invariant type variable for the result data of a model.

We need to use an invariant typevar for `OutputValidator` and `OutputValidatorFunc` because the output data type is used
in both the input and output of a `OutputValidatorFunc`. This can theoretically lead to some issues assuming that types
possessing OutputValidator's are covariant in the result data type, but in practice this is rarely an issue, and
changing it would have negative consequences for the ergonomics of the library.

At some point, it may make sense to change the input to OutputValidatorFunc to be `Any` or `object` as doing that would
resolve these potential variance issues.
"""
OutputDataT = TypeVar('OutputDataT', default=str, covariant=True)
"""Covariant type variable for the result data type of a run."""

OutputValidatorFunc = Union[
    Callable[[RunContext[AgentDepsT], OutputDataT_inv], OutputDataT_inv],
    Callable[[RunContext[AgentDepsT], OutputDataT_inv], Awaitable[OutputDataT_inv]],
    Callable[[OutputDataT_inv], OutputDataT_inv],
    Callable[[OutputDataT_inv], Awaitable[OutputDataT_inv]],
]
"""
A function that always takes and returns the same type of data (which is the result type of an agent run), and:

* may or may not take [`RunContext`][pydantic_ai.tools.RunContext] as a first argument
* may or may not be async

Usage `OutputValidatorFunc[AgentDepsT, T]`.
"""


DEFAULT_OUTPUT_TOOL_NAME = 'final_result'
DEFAULT_OUTPUT_TOOL_DESCRIPTION = 'The final response which ends this conversation'
DEFAULT_MANUAL_JSON_PROMPT = dedent(  # TODO: Move to ModelProfile
    """
    Always respond with a JSON object matching this description and schema:

    {description}

    {schema}

    Don't include any text or Markdown fencing before or after.
    """
)


@dataclass
class OutputValidator(Generic[AgentDepsT, OutputDataT_inv]):
    function: OutputValidatorFunc[AgentDepsT, OutputDataT_inv]
    _takes_ctx: bool = field(init=False)
    _is_async: bool = field(init=False)

    def __post_init__(self):
        self._takes_ctx = len(inspect.signature(self.function).parameters) > 1
        self._is_async = inspect.iscoroutinefunction(self.function)

    async def validate(
        self,
        result: T,
        tool_call: _messages.ToolCallPart | None,
        run_context: RunContext[AgentDepsT],
    ) -> T:
        """Validate a result but calling the function.

        Args:
            result: The result data after Pydantic validation the message content.
            tool_call: The original tool call message, `None` if there was no tool call.
            run_context: The current run context.

        Returns:
            Result of either the validated result data (ok) or a retry message (Err).
        """
        if self._takes_ctx:
            ctx = run_context.replace_with(tool_name=tool_call.tool_name if tool_call else None)
            args = ctx, result
        else:
            args = (result,)

        try:
            if self._is_async:
                function = cast(Callable[[Any], Awaitable[T]], self.function)
                result_data = await function(*args)
            else:
                function = cast(Callable[[Any], T], self.function)
                result_data = await _utils.run_in_executor(function, *args)
        except ModelRetry as r:
            m = _messages.RetryPromptPart(content=r.message)
            if tool_call is not None:
                m.tool_name = tool_call.tool_name
                m.tool_call_id = tool_call.tool_call_id
            raise ToolRetryError(m) from r
        else:
            return result_data


class ToolRetryError(Exception):
    """Internal exception used to signal a `ToolRetry` message should be returned to the LLM."""

    def __init__(self, tool_retry: _messages.RetryPromptPart):
        self.tool_retry = tool_retry
        super().__init__()


@dataclass(init=False)
class ToolOutput(Generic[OutputDataT]):
    """Marker class to use tools for outputs, and customize the tool."""

    output_type: OutputTypeOrFunction[OutputDataT]  # TODO: Allow list of types instead of unions?
    name: str | None
    description: str | None
    max_retries: int | None
    strict: bool | None

    def __init__(
        self,
        type_: OutputTypeOrFunction[OutputDataT],
        *,
        name: str | None = None,
        description: str | None = None,
        max_retries: int | None = None,
        strict: bool | None = None,
    ):
        self.output_type = type_
        self.name = name
        self.description = description
        self.max_retries = max_retries
        self.strict = strict


@dataclass
class TextOutput(Generic[OutputDataT]):
    """Marker class to use text output for outputs."""

    output_type: (
        Callable[[RunContext, str], Awaitable[OutputDataT] | OutputDataT]
        | Callable[[str], Awaitable[OutputDataT] | OutputDataT]
    )


@dataclass(init=False)
class JSONSchemaOutput(Generic[OutputDataT]):
    """Marker class to use JSON schema output for outputs."""

    output_types: Sequence[OutputTypeOrFunction[OutputDataT]]
    name: str | None
    description: str | None
    strict: bool | None

    def __init__(
        self,
        type_: OutputTypeOrFunction[OutputDataT] | Sequence[OutputTypeOrFunction[OutputDataT]],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ):
        self.output_types = flatten_output_types(type_)
        self.name = name
        self.description = description
        self.strict = strict


class ManualJSONOutput(Generic[OutputDataT]):
    """Marker class to use manual JSON mode for outputs."""

    output_types: Sequence[OutputTypeOrFunction[OutputDataT]]
    name: str | None
    description: str | None

    def __init__(
        self,
        type_: OutputTypeOrFunction[OutputDataT] | Sequence[OutputTypeOrFunction[OutputDataT]],
        *,
        name: str | None = None,
        description: str | None = None,
    ):
        self.output_types = flatten_output_types(type_)
        self.name = name
        self.description = description


T_co = TypeVar('T_co', covariant=True)

OutputTypeOrFunction = TypeAliasType(
    'OutputTypeOrFunction', Union[type[T_co], Callable[..., Awaitable[T_co] | T_co]], type_params=(T_co,)
)
OutputType = TypeAliasType(
    'OutputType',
    Union[
        OutputTypeOrFunction[T_co],
        ToolOutput[T_co],
        TextOutput[T_co],
        Sequence[Union[OutputTypeOrFunction[T_co], ToolOutput[T_co], TextOutput[T_co]]],
        JSONSchemaOutput[T_co],
        ManualJSONOutput[T_co],
    ],
    type_params=(T_co,),
)

# TODO: Add `json_object` for old OpenAI models, or rename `json_schema` to `json` and choose automatically, relying on Pydantic validation
OutputMode = Literal['text', 'tool', 'tool_or_text', 'json_schema', 'manual_json']


@dataclass(init=False)
class OutputSchema(Generic[OutputDataT]):
    """Model the final output from an agent run.

    Similar to `Tool` but for the final output of running an agent.
    """

    mode: OutputMode | None = None
    text_output_schema: (
        OutputObjectSchema[OutputDataT] | OutputUnionSchema[OutputDataT] | OutputTextSchema[OutputDataT] | None
    ) = None
    tools: dict[str, OutputTool[OutputDataT]] = field(default_factory=dict)

    def __init__(
        self,
        output_type: OutputType[OutputDataT],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ):
        """Build an OutputSchema dataclass from an output type."""
        self.mode = None
        self.text_output_schema = None
        self.tools = {}

        if output_type is str:
            self.mode = 'text'
            self.text_output_schema = OutputTextSchema(output_type)
            return

        if isinstance(output_type, JSONSchemaOutput):
            self.mode = 'json_schema'
            self.text_output_schema = self._build_text_output_schema(
                output_type.output_types,
                name=output_type.name,
                description=output_type.description,
                strict=output_type.strict,
            )
            return

        if isinstance(output_type, ManualJSONOutput):
            self.mode = 'manual_json'
            self.text_output_schema = self._build_text_output_schema(
                output_type.output_types, name=output_type.name, description=output_type.description
            )
            return

        text_outputs: Sequence[type[str] | TextOutput[OutputDataT]] = []
        tool_outputs: Sequence[ToolOutput[OutputDataT]] = []
        other_outputs: Sequence[OutputTypeOrFunction[OutputDataT]] = []
        for output_type_or_marker in flatten_output_types(output_type):
            if output_type_or_marker is str:
                text_outputs.append(cast(type[str], output_type_or_marker))
            elif isinstance(output_type_or_marker, TextOutput):
                text_outputs.append(output_type_or_marker)
            elif isinstance(output_type_or_marker, ToolOutput):
                tool_outputs.append(output_type_or_marker)
            else:
                other_outputs.append(output_type_or_marker)

        self.tools = self._build_tools(tool_outputs + other_outputs, name=name, description=description, strict=strict)

        if len(text_outputs) > 0:
            if len(text_outputs) > 1:
                raise UserError('Only one text output is allowed')
            text_output = text_outputs[0]

            self.mode = 'text'
            if len(self.tools) > 0:
                self.mode = 'tool_or_text'

            if isinstance(text_output, TextOutput):
                self.text_output_schema = OutputTextSchema(text_output.output_type)
            elif text_output is str:
                self.text_output_schema = cast(OutputTextSchema[OutputDataT], OutputTextSchema(text_output))
        elif len(tool_outputs) > 0:
            self.mode = 'tool'
        else:
            self.text_output_schema = self._build_text_output_schema(
                other_outputs, name=name, description=description, strict=strict
            )

    @staticmethod
    def _build_tools(
        outputs: list[OutputTypeOrFunction[OutputDataT] | ToolOutput[OutputDataT]],
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> dict[str, OutputTool[OutputDataT]]:
        tools: dict[str, OutputTool[OutputDataT]] = {}

        default_name = name or DEFAULT_OUTPUT_TOOL_NAME
        default_description = description
        default_strict = strict

        multiple = len(outputs) > 1
        for output in outputs:
            name = None
            description = None
            strict = None
            if isinstance(output, ToolOutput):
                output_type = output.output_type
                # do we need to error on conflicts here? (DavidM): If this is internal maybe doesn't matter, if public, use overloads
                name = output.name
                description = output.description
                strict = output.strict
            else:
                output_type = output

            if name is None:
                name = default_name
                if multiple:
                    name += f'_{output_type.__name__}'

            i = 1
            original_name = name
            while name in tools:
                i += 1
                name = f'{original_name}_{i}'

            description = description or default_description
            if strict is None:
                strict = default_strict

            parameters_schema = OutputObjectSchema(output_type=output_type, description=description, strict=strict)
            tools[name] = OutputTool(name=name, parameters_schema=parameters_schema, multiple=multiple)

        return tools

    @staticmethod
    def _build_text_output_schema(
        outputs: Sequence[OutputTypeOrFunction[OutputDataT]],
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> OutputObjectSchema[OutputDataT] | OutputUnionSchema[OutputDataT] | None:
        if len(outputs) == 0:
            return None

        outputs = flatten_output_types(outputs)
        if len(outputs) == 1:
            return OutputObjectSchema(output_type=outputs[0], name=name, description=description, strict=strict)

        return OutputUnionSchema(output_types=outputs, name=name, description=description, strict=strict)

    @property
    def allow_text_output(self) -> Literal['plain', 'json', False]:
        """Whether the model allows text output."""
        if self.mode == 'tool':
            return False
        if self.mode in ('text', 'tool_or_text'):
            return 'plain'
        return 'json'

    def is_mode_supported(self, profile: ModelProfile) -> bool:
        """Whether the model supports the output mode."""
        mode = self.mode
        if mode in ('text', 'manual_json'):
            return True
        if self.mode == 'tool_or_text':
            mode = 'tool'
        return mode in profile.output_modes

    def find_named_tool(
        self, parts: Iterable[_messages.ModelResponsePart], tool_name: str
    ) -> tuple[_messages.ToolCallPart, OutputTool[OutputDataT]] | None:
        """Find a tool that matches one of the calls, with a specific name."""
        for part in parts:  # pragma: no branch
            if isinstance(part, _messages.ToolCallPart):  # pragma: no branch
                if part.tool_name == tool_name:
                    return part, self.tools[tool_name]

    def find_tool(
        self,
        parts: Iterable[_messages.ModelResponsePart],
    ) -> Iterator[tuple[_messages.ToolCallPart, OutputTool[OutputDataT]]]:
        """Find a tool that matches one of the calls."""
        for part in parts:
            if isinstance(part, _messages.ToolCallPart):  # pragma: no branch
                if result := self.tools.get(part.tool_name):
                    yield part, result

    def tool_names(self) -> list[str]:
        """Return the names of the tools."""
        return list(self.tools.keys())

    def tool_defs(self) -> list[ToolDefinition]:
        """Get tool definitions to register with the model."""
        return [t.tool_def for t in self.tools.values()]

    async def process(
        self,
        text: str,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        """Validate an output message.

        Args:
            text: The output text to validate.
            run_context: The current run context.
            allow_partial: If true, allow partial validation.
            wrap_validation_errors: If true, wrap the validation errors in a retry message.

        Returns:
            Either the validated output data (left) or a retry message (right).
        """
        assert self.allow_text_output is not False
        assert self.text_output_schema is not None

        # TODO: Strip Markdown fences?
        return await self.text_output_schema.process(
            text, run_context, allow_partial=allow_partial, wrap_validation_errors=wrap_validation_errors
        )


@dataclass
class OutputObjectDefinition:
    name: str
    json_schema: ObjectJsonSchema
    description: str | None = None
    strict: bool | None = None

    @property
    def manual_json_instructions(self) -> str:
        """Get instructions for model to output manual JSON matching the schema."""
        # TODO: Move to ModelProfile so it can be tweaked
        description = ': '.join([v for v in [self.name, self.description] if v])
        return DEFAULT_MANUAL_JSON_PROMPT.format(schema=json.dumps(self.json_schema), description=description)


@dataclass(init=False)
class OutputUnionDataEntry:
    kind: str
    data: dict[str, Any]


@dataclass(init=False)
class OutputUnionData:
    result: OutputUnionDataEntry


# TODO: Better class naming
@dataclass(init=False)
class OutputUnionSchema(Generic[OutputDataT]):
    object_def: OutputObjectDefinition
    _root_object_schema: OutputObjectSchema[OutputUnionData]
    _object_schemas: dict[str, OutputObjectSchema[OutputDataT]]

    def __init__(
        self,
        output_types: Sequence[OutputTypeOrFunction[OutputDataT]],
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ):
        self._object_schemas = {}
        # TODO: Ensure keys are unique
        self._object_schemas = {
            output_type.__name__: OutputObjectSchema(output_type=output_type) for output_type in output_types
        }

        self._root_object_schema = OutputObjectSchema(output_type=OutputUnionData)

        # TODO: Account for conflicting $defs and $refs
        json_schema = {
            'type': 'object',
            'properties': {
                'result': {
                    'anyOf': [
                        {
                            'type': 'object',
                            'properties': {
                                'kind': {
                                    'const': name,
                                },
                                'data': object_schema.object_def.json_schema,  # TODO: Pop description here?
                            },
                            'description': object_schema.object_def.description or name,  # TODO: Better description
                            'required': ['kind', 'data'],
                            'additionalProperties': False,
                        }
                        for name, object_schema in self._object_schemas.items()
                    ],
                }
            },
            'required': ['result'],
            'additionalProperties': False,
        }

        self.object_def = OutputObjectDefinition(
            name=name or DEFAULT_OUTPUT_TOOL_NAME,
            description=description or DEFAULT_OUTPUT_TOOL_DESCRIPTION,
            json_schema=json_schema,
            strict=strict,
        )

    async def process(
        self,
        data: str | dict[str, Any] | None,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        # TODO: Error handling?
        result = await self._root_object_schema.process(
            data, run_context, allow_partial=allow_partial, wrap_validation_errors=wrap_validation_errors
        )

        result = result.result
        kind = result.kind
        data = result.data
        try:
            object_schema = self._object_schemas[kind]
        except KeyError as e:
            raise ToolRetryError(_messages.RetryPromptPart(content=f'Invalid kind: {kind}')) from e

        return await object_schema.process(
            data, run_context, allow_partial=allow_partial, wrap_validation_errors=wrap_validation_errors
        )


@dataclass(init=False)
class OutputObjectSchema(Generic[OutputDataT]):
    object_def: OutputObjectDefinition
    outer_typed_dict_key: str | None = None
    _validator: SchemaValidator
    _function_schema: _function_schema.FunctionSchema | None = None

    def __init__(
        self,
        output_type: OutputTypeOrFunction[OutputDataT],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ):
        if inspect.isfunction(output_type) or inspect.ismethod(output_type):
            self._function_schema = _function_schema.function_schema(output_type, GenerateToolJsonSchema)
            self._validator = self._function_schema.validator
            json_schema = self._function_schema.json_schema
            json_schema['description'] = self._function_schema.description
        else:
            type_adapter: TypeAdapter[Any]
            if _utils.is_model_like(output_type):
                type_adapter = TypeAdapter(output_type)
            else:
                self.outer_typed_dict_key = 'response'
                response_data_typed_dict = TypedDict(  # noqa: UP013
                    'response_data_typed_dict',
                    {'response': cast(type[OutputDataT], output_type)},  # pyright: ignore[reportInvalidTypeForm]
                )
                type_adapter = TypeAdapter(response_data_typed_dict)

            # Really a PluggableSchemaValidator, but it's API-compatible
            self._validator = cast(SchemaValidator, type_adapter.validator)
            json_schema = _utils.check_object_json_schema(
                type_adapter.json_schema(schema_generator=GenerateToolJsonSchema)
            )

            if self.outer_typed_dict_key:
                # including `response_data_typed_dict` as a title here doesn't add anything and could confuse the LLM
                json_schema.pop('title')

        if json_schema_description := json_schema.pop('description', None):
            if description is None:
                description = json_schema_description
            else:
                description = f'{description}. {json_schema_description}'

        self.object_def = OutputObjectDefinition(
            name=name or getattr(output_type, '__name__', DEFAULT_OUTPUT_TOOL_NAME),
            description=description,
            json_schema=json_schema,
            strict=strict,
        )

    async def process(
        self,
        data: str | dict[str, Any] | None,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        """Process an output message, performing validation and (if necessary) calling the output function.

        Args:
            data: The output data to validate.
            run_context: The current run context.
            allow_partial: If true, allow partial validation.
            wrap_validation_errors: If true, wrap the validation errors in a retry message.

        Returns:
            Either the validated output data (left) or a retry message (right).
        """
        try:
            pyd_allow_partial: Literal['off', 'trailing-strings'] = 'trailing-strings' if allow_partial else 'off'
            if isinstance(data, str):
                output = self._validator.validate_json(data or '{}', allow_partial=pyd_allow_partial)
            else:
                output = self._validator.validate_python(data or {}, allow_partial=pyd_allow_partial)
        except ValidationError as e:
            if wrap_validation_errors:
                m = _messages.RetryPromptPart(
                    content=e.errors(include_url=False),
                )
                raise ToolRetryError(m) from e
            else:
                raise

        if k := self.outer_typed_dict_key:
            output = output[k]

        if self._function_schema:
            try:
                output = await self._function_schema.call(output, run_context)
            except ModelRetry as r:
                if wrap_validation_errors:
                    m = _messages.RetryPromptPart(
                        content=r.message,
                    )
                    raise ToolRetryError(m) from r
                else:
                    raise

        return output


@dataclass(init=False)
class OutputTextSchema(Generic[OutputDataT]):
    _function_schema: _function_schema.FunctionSchema | None = None
    _str_argument_name: str | None = None

    def __init__(
        self,
        output_type: type[OutputDataT]
        | Callable[[RunContext[AgentDepsT], str], Awaitable[OutputDataT] | OutputDataT]
        | Callable[[str], Awaitable[OutputDataT] | OutputDataT] = str,
    ):
        if inspect.isfunction(output_type) or inspect.ismethod(output_type):
            self._function_schema = _function_schema.function_schema(output_type, GenerateToolJsonSchema)
            arguments_schema = self._function_schema.json_schema.get('properties', {})
            argument_name = next(iter(arguments_schema.keys()), None)
            if argument_name and arguments_schema.get(argument_name, {}).get('type') == 'string':
                self._str_argument_name = argument_name
                return
        elif output_type is str:
            return

        raise ValueError('OutputTextSchema must take the `str` type or a function taking a `str`')

    @property
    def object_def(self) -> None:
        return None

    async def process(
        self,
        data: str,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        output = data

        if self._function_schema and self._str_argument_name:
            try:
                output = await self._function_schema.call({self._str_argument_name: output}, run_context)
            except ModelRetry as r:
                if wrap_validation_errors:
                    m = _messages.RetryPromptPart(
                        content=r.message,
                    )
                    raise ToolRetryError(m) from r
                else:
                    raise

        return cast(OutputDataT, output)


@dataclass(init=False)
class OutputTool(Generic[OutputDataT]):
    parameters_schema: OutputObjectSchema[OutputDataT]
    tool_def: ToolDefinition

    def __init__(self, *, name: str, parameters_schema: OutputObjectSchema[OutputDataT], multiple: bool):
        self.parameters_schema = parameters_schema
        definition = parameters_schema.object_def

        description = definition.description
        if not description:
            description = DEFAULT_OUTPUT_TOOL_DESCRIPTION
            if multiple:
                description = f'{definition.name}: {description}'

        self.tool_def = ToolDefinition(
            name=name,
            description=description,
            parameters_json_schema=definition.json_schema,
            strict=definition.strict,
            outer_typed_dict_key=parameters_schema.outer_typed_dict_key,
        )

    async def process(
        self,
        tool_call: _messages.ToolCallPart,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        """Process an output message.

        Args:
            tool_call: The tool call from the LLM to validate.
            run_context: The current run context.
            allow_partial: If true, allow partial validation.
            wrap_validation_errors: If true, wrap the validation errors in a retry message.

        Returns:
            Either the validated output data (left) or a retry message (right).
        """
        try:
            output = await self.parameters_schema.process(
                tool_call.args, run_context, allow_partial=allow_partial, wrap_validation_errors=False
            )
        except ValidationError as e:
            if wrap_validation_errors:
                m = _messages.RetryPromptPart(
                    tool_name=tool_call.tool_name,
                    content=e.errors(include_url=False, include_context=False),
                    tool_call_id=tool_call.tool_call_id,
                )
                raise ToolRetryError(m) from e
            else:
                raise  # pragma: lax no cover
        except ModelRetry as r:
            if wrap_validation_errors:
                m = _messages.RetryPromptPart(
                    tool_name=tool_call.tool_name,
                    content=r.message,
                    tool_call_id=tool_call.tool_call_id,
                )
                raise ToolRetryError(m) from r
            else:
                raise  # pragma: lax no cover
        else:
            return output


def get_union_args(tp: Any) -> tuple[Any, ...]:
    """Extract the arguments of a Union type if `output_type` is a union, otherwise return an empty tuple."""
    if typing_objects.is_typealiastype(tp):
        tp = tp.__value__

    origin = get_origin(tp)
    if is_union_origin(origin):
        return get_args(tp)
    else:
        return ()


def flatten_output_types(output_type: T | Sequence[T]) -> list[T]:
    output_types: Sequence[T]
    if isinstance(output_type, Sequence):
        output_types = output_type
    else:
        output_types = (output_type,)

    output_types_flat: list[T] = []
    for output_type in output_types:
        if union_types := get_union_args(output_type):
            output_types_flat.extend(union_types)
        else:
            output_types_flat.append(output_type)
    return output_types_flat
