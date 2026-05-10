#!/usr/bin/env python3
"""
Tool definitions for testchat — mock tools to test model tool-calling capability.

These tools are NEVER executed. They exist solely to:
1. Test if a model can decide which tool to call
2. Test if a model generates well-formed JSON arguments
3. Provide mock responses back to the model to test multi-turn tool use

Each tool has a mock response that simulates realistic API behavior.
"""

# Tool definitions in OpenAI function-calling format
MOCK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city. Returns temperature, conditions, and humidity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'São Paulo', 'Tokyo'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit. Defaults to celsius."
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Supports +, -, *, /, **, parentheses, and basic functions (sqrt, sin, cos, log).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate, e.g. '2 + 2' or 'sqrt(144)'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current date and time for a given timezone or city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone name or city name, e.g. 'America/Sao_Paulo', 'Europe/London', 'Tokyo'"
                    }
                },
                "required": ["timezone"]
            }
        }
    }
]

# Mock responses — deterministic but realistic
# The model never sees these directly; they're returned as tool results
MOCK_RESPONSES = {
    "get_weather": {
        # City-normalized responses (strtolower, remove accents for matching)
        "sao paulo": {"city": "São Paulo", "temperature": 26, "unit": "celsius", "conditions": "Partly cloudy", "humidity": 68},
        "rio de janeiro": {"city": "Rio de Janeiro", "temperature": 32, "unit": "celsius", "conditions": "Sunny", "humidity": 72},
        "diamantina": {"city": "Diamantina", "temperature": 22, "unit": "celsius", "conditions": "Clear skies", "humidity": 55},
        "tokyo": {"city": "Tokyo", "temperature": 18, "unit": "celsius", "conditions": "Overcast", "humidity": 60},
        "london": {"city": "London", "temperature": 12, "unit": "celsius", "conditions": "Rainy", "humidity": 85},
        "new york": {"city": "New York", "temperature": 15, "unit": "celsius", "conditions": "Cloudy", "humidity": 65},
        "paris": {"city": "Paris", "temperature": 14, "unit": "celsius", "conditions": "Light rain", "humidity": 78},
        "berlin": {"city": "Berlin", "temperature": 8, "unit": "celsius", "conditions": "Foggy", "humidity": 90},
    },
    "calculator": {
        # Evaluated dynamically — this is a safety valve for known expressions
        "2+2": "4",
        "2 + 2": "4",
        "sqrt(144)": "12",
        "10 * 5": "50",
    },
    "get_time": {
        "america/sao paulo": "2025-05-09 22:00:00 BRT (UTC-3)",
        "america/sao_paulo": "2025-05-09 22:00:00 BRT (UTC-3)",
        "europe/london": "2025-05-10 02:00:00 BST (UTC+1)",
        "asia/tokyo": "2025-05-10 10:00:00 JST (UTC+9)",
        "us/eastern": "2025-05-09 21:00:00 EDT (UTC-4)",
    }
}


def get_mock_response(tool_name: str, arguments: dict) -> str:
    """Generate a mock response for a tool call.
    
    For incorrect/missing arguments, returns a didactic error message
    explaining what went wrong, so the model can learn from its mistake.
    
    For valid arguments, returns a realistic mock data response.
    """
    if tool_name == "get_weather":
        city = arguments.get("city", "").strip().lower()
        unit = arguments.get("unit", "celsius")
        
        if not city:
            return "Error: The 'city' parameter is required. Please provide a city name like 'São Paulo' or 'Tokyo'."
        
        # Try to find a matching city (normalize accents)
        import unicodedata
        def normalize(s):
            return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii').lower().strip()
        
        city_normalized = normalize(city)
        for key, data in MOCK_RESPONSES["get_weather"].items():
            if normalize(key) == city_normalized:
                # Convert unit if needed
                result = dict(data)
                if unit == "fahrenheit" and result["unit"] == "celsius":
                    result["temperature"] = round(result["temperature"] * 9/5 + 32, 1)
                    result["unit"] = "fahrenheit"
                import json
                return json.dumps(result, ensure_ascii=False)
        
        # Unknown city — return a plausible mock
        import json
        return json.dumps({
            "city": city,
            "temperature": 20,
            "unit": unit,
            "conditions": "Partly cloudy",
            "humidity": 50
        }, ensure_ascii=False)
    
    elif tool_name == "calculator":
        expression = arguments.get("expression", "").strip()
        
        if not expression:
            return "Error: The 'expression' parameter is required. Example: '2 + 2' or 'sqrt(144)'."
        
        # Safety: only allow safe math expressions
        import re
        safe_expr = re.sub(r'[^0-9+\-*/().%\s\*]', '', expression.replace('**', '^POW^').replace('^POW^', '**'))
        # Replace common functions with math module equivalents
        safe_expr = safe_expr.replace('sqrt', 'math.sqrt')
        safe_expr = safe_expr.replace('sin', 'math.sin')
        safe_expr = safe_expr.replace('cos', 'math.cos')
        safe_expr = safe_expr.replace('log', 'math.log')
        
        # Check for known expressions first
        if expression in MOCK_RESPONSES["calculator"]:
            return MOCK_RESPONSES["calculator"][expression]
        
        try:
            import math
            result = eval(safe_expr, {"__builtins__": {}}, {"math": math})
            return str(result)
        except Exception as e:
            return f"Error: Could not evaluate expression '{expression}'. {type(e).__name__}: {e}. Please provide a valid mathematical expression."
    
    elif tool_name == "get_time":
        timezone = arguments.get("timezone", "").strip()
        
        if not timezone:
            return "Error: The 'timezone' parameter is required. Please provide a timezone like 'America/Sao_Paulo' or a city name."
        
        # Normalize
        tz_lower = timezone.lower().replace(" ", "_")
        
        for key, value in MOCK_RESPONSES["get_time"].items():
            if key.lower() == tz_lower:
                return value
        
        # Generic response for unknown timezones
        return f"{timezone}: 2025-05-09 22:00:00 (UTC-3)"
    
    else:
        return f"Error: Unknown tool '{tool_name}'. Available tools: get_weather, calculator, get_time."