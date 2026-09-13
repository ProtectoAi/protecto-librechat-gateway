import json
from typing import List,Dict,Any,Tuple
from .config import logger

def openai_messages_to_responses(
        messages: List[Dict[str, Any]],
) -> Tuple[list, str | None]:
    """
    Convert Chat-Completions-style masked history into Responses API input.

      system/developer     -> instructions
      user/assistant       -> conversation messages
      assistant.tool_calls -> function_call input items
      tool                 -> function_call_output input items
    """
    instructions = []
    response_input = []

    for msg in messages:
        role = msg.get("role")

        if role in {"system", "developer"}:
            content = msg.get("content", "")
            if content:
                instructions.append(content)
            continue

        if role in {"user", "assistant"}:
            content = msg.get("content")
            if content:
                response_input.append({"role": role, "content": content})
            if role == "assistant":
                for tool_call in msg.get("tool_calls") or []:
                    if tool_call.get("type") != "function":
                        continue
                    function = tool_call.get("function") or {}
                    call_id = tool_call.get("id")
                    if not call_id or not function.get("name"):
                        continue
                    arguments = function.get("arguments", "{}")
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments, ensure_ascii=False)
                    response_input.append({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": function["name"],
                        "arguments": arguments,
                    })
            continue

        if role == "tool":
            call_id = msg.get("tool_call_id")
            if not call_id:
                logger.warning("Ignoring tool message without tool_call_id")
                continue
            output = msg.get("content", "")
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            response_input.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            })
            continue

        logger.warning("Ignoring unsupported OpenAI history role: %s", role)

    return response_input, ("\n\n".join(instructions) if instructions else None)


def librechat_tools_to_responses(tools: List[Dict[str, Any]] | None) -> list:
    converted = []
    for tool in tools or []:
        if tool.get("type") != "function":
            logger.warning(
                "Ignoring unsupported LibreChat tool type: %s", tool.get("type"),
            )
            continue
        fn = tool.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        item = {
            "type": "function",
            "name": name,
            "parameters": fn.get(
                "parameters",
                {"type": "object", "properties": {}, "additionalProperties": False},
            ),
        }
        if fn.get("description"):
            item["description"] = fn["description"]
        if "strict" in fn:
            item["strict"] = fn["strict"]
        converted.append(item)
    return converted


def extract_openai_tool_calls(response_json: dict) -> list:
    tool_calls = []
    for item in response_json.get("output", []):
        if item.get("type") != "function_call":
            continue
        call_id = item.get("call_id") or item.get("id")
        name = item.get("name")
        arguments = item.get("arguments", "{}")
        if not call_id or not name:
            continue
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        tool_calls.append({
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        })
    return tool_calls


def extract_openai_response_text(response_json: dict) -> str:
    parts = []
    for item in response_json.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "".join(parts)



def build_openai_payload(model:str,messages:list[dict],tools:list|None,tool_choice:any,stream:bool)->dict:
    response_input,instructions=openai_messages_to_responses(messages)
    payload={"model":model,"input":response_input,"stream":stream}
    if instructions: payload["instructions"]=instructions
    if tools:
        payload["tools"]=librechat_tools_to_responses(tools)
        payload["parallel_tool_calls"]=True
    if tool_choice is not None:
        if isinstance(tool_choice,dict) and tool_choice.get("type")=="function" and (tool_choice.get("function") or {}).get("name"):
            payload["tool_choice"]={"type":"function","name":tool_choice["function"]["name"]}
        else: payload["tool_choice"]=tool_choice
    return payload
