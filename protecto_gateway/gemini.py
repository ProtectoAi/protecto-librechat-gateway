"""Gemini Interactions API adapter."""
import json
from typing import Any
from .config import GEMINI_SCHEMA_ALLOWED_KEYS,GEMINI_ALLOWED_FORMATS,logger
from .state import lookup_gemini_state

def extract_gemini_interaction_text(response_json:dict)->str:
    return "".join(part.get("text","") for step in response_json.get("steps",[]) if step.get("type")=="model_output" for part in step.get("content",[]) if part.get("type")=="text")

def sanitize_gemini_schema(schema:Any)->Any:
    if isinstance(schema,list): return [sanitize_gemini_schema(x) for x in schema]
    if not isinstance(schema,dict): return schema
    cleaned={}
    for key,value in schema.items():
        if key not in GEMINI_SCHEMA_ALLOWED_KEYS: continue
        if key=="properties" and isinstance(value,dict): cleaned[key]={k:sanitize_gemini_schema(v) for k,v in value.items()}
        elif key in {"items","anyOf"}: cleaned[key]=sanitize_gemini_schema(value)
        else: cleaned[key]=value
    fmt=cleaned.get("format")
    if fmt is not None and fmt not in GEMINI_ALLOWED_FORMATS.get(cleaned.get("type"),set()): cleaned.pop("format",None)
    if cleaned.get("type")=="object" and "properties" not in cleaned: cleaned["properties"]={}
    if isinstance(cleaned.get("properties"),dict) and "required" in cleaned:
        cleaned["required"]=[x for x in cleaned["required"] if x in cleaned["properties"]]
        if not cleaned["required"]: cleaned.pop("required")
    return cleaned

def librechat_tools_to_gemini(tools:list|None)->list:
    result=[]
    for tool in tools or []:
        if tool.get("type")!="function": continue
        fn=tool.get("function") or {}; name=fn.get("name")
        if not name: continue
        item={"type":"function","name":name,"parameters":sanitize_gemini_schema(fn.get("parameters",{"type":"object","properties":{}}))}
        if fn.get("description"): item["description"]=fn["description"]
        result.append(item)
    return result

def build_gemini_payload(model:str,messages:list[dict],tools:list|None,stream:bool,gemini_state:dict|None=None)->dict:
    system_parts=[]; input_steps=[]
    call_names={}
    for msg in messages:
        for tc in msg.get("tool_calls") or []:
            fn=tc.get("function") or {}
            if tc.get("id") and fn.get("name"): call_names[tc["id"]]=fn["name"]

    state=gemini_state or lookup_gemini_state(messages)
    previous_id=(state or {}).get("interaction_id")
    if previous_id:
        current_call_ids=(state or {}).get("call_ids", set())
        for msg in messages:
            if msg.get("role")!="tool": continue
            call_id=msg.get("tool_call_id")
            if not call_id or (current_call_ids and call_id not in current_call_ids): continue
            provider_call_id=(state or {}).get("provider_ids", {}).get(call_id, call_id)
            result=msg.get("content","")
            if not isinstance(result,str): result=json.dumps(result,ensure_ascii=False)
            step={"type":"function_result","call_id":provider_call_id,"result":[{"type":"text","text":result}]}
            name=msg.get("name") or call_names.get(call_id)
            if name: step["name"]=name
            input_steps.append(step)
    else:
        for msg in messages:
            role=msg.get("role")
            if role in {"system","developer"}:
                if msg.get("content"): system_parts.append(msg["content"])
            elif role=="user" and msg.get("content"):
                input_steps.append({"type":"user_input","content":[{"type":"text","text":msg["content"]}]})
            elif role=="assistant":
                if msg.get("content"):
                    input_steps.append({"type":"model_output","content":[{"type":"text","text":msg["content"]}]})
                for tc in msg.get("tool_calls") or []:
                    fn=tc.get("function") or {}
                    if tc.get("id") and fn.get("name"):
                        args=fn.get("arguments","{}")
                        if isinstance(args,str):
                            try: args=json.loads(args)
                            except json.JSONDecodeError: args={"raw":args}
                        input_steps.append({"type":"function_call","id":tc["id"],"name":fn["name"],"arguments":args})
            elif role=="tool":
                call_id=msg.get("tool_call_id")
                if call_id:
                    result=msg.get("content","")
                    if not isinstance(result,str): result=json.dumps(result,ensure_ascii=False)
                    step={"type":"function_result","call_id":call_id,"result":[{"type":"text","text":result}]}
                    if msg.get("name") or call_names.get(call_id): step["name"]=msg.get("name") or call_names[call_id]
                    input_steps.append(step)

    payload={"model":model,"input":input_steps,"stream":stream,"store":True}
    logger.info(
        "[GEMINI PAYLOAD DEBUG] previous_id=%s input_steps=%d tools=%d system=%s",
        previous_id, len(input_steps), len(tools or []), bool(system_parts),
    )
    for i, step in enumerate(input_steps):
        logger.info(
            "[GEMINI PAYLOAD DEBUG] INPUT[%d] type=%s call_id=%s name=%s keys=%s",
            i, step.get("type"), step.get("call_id"), step.get("name"), sorted(step.keys()),
        )
    if previous_id:
        payload["previous_interaction_id"] = previous_id
        logger.info("[GEMINI PAYLOAD DEBUG] using previous_interaction_id=%s", previous_id)

    # Gemini only carries the conversation through previous_interaction_id.
    # Tool declarations and system instructions are scoped to *each* new
    # Interaction, so omitting tools here makes a continuation unable to make
    # another tool call (the common multi-tool / multi-round failure mode).
    if tools:
        payload["tools"] = librechat_tools_to_gemini(tools)
    if system_parts:
        payload["system_instruction"] = "\n\n".join(system_parts)
    return payload

def extract_gemini_tool_calls(response_json:dict)->list:
    result=[]
    for step in response_json.get("steps",[]):
        if step.get("type")!="function_call": continue
        call_id=step.get("id") or step.get("call_id"); name=step.get("name")
        if not call_id or not name: continue
        args=step.get("arguments",{})
        if not isinstance(args,str): args=json.dumps(args,ensure_ascii=False)
        result.append({"id":call_id,"type":"function","function":{"name":name,"arguments":args}})
    return result
