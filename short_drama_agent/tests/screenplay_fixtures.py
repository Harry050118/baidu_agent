import json

from src.skills.screenplay.schemas import GenerationConstraints


def valid_screenplay_dict():
    return {
        "title": "最后一课",
        "genre": "悬疑",
        "logline": "学生追查消失的录音。",
        "theme": "信任",
        "opening_hook": "失踪者的声音突然响起。",
        "main_conflict": "林夏必须在老师阻挠下找到真相。",
        "reversal": "录音来自未来。",
        "emotional_payoff": "林夏选择公开真相。",
        "characters": [
            {"name": "林夏", "role": "学生", "personality": "执着", "goal": "找到真相"}
        ],
        "scenes": [
            {
                "scene_id": scene_id,
                "location": "教室",
                "time": "夜",
                "purpose": "发现线索",
                "conflict": "有人试图阻止调查",
                "visual_action": "林夏推开门，看见桌上的录音笔。",
                "dialogues": [{"character": "林夏", "line": "真相就在这里。"}],
                "estimated_seconds": 60,
                "ending_hook": "门外传来脚步声。" if scene_id == 1 else None,
            }
            for scene_id in range(1, 4)
        ],
        "ending": "真相被公开。",
    }


def valid_screenplay_json():
    return json.dumps(valid_screenplay_dict(), ensure_ascii=False)


def skill_inputs():
    return {
        "story_plan": {"genre": "悬疑", "outline": "学生追查录音"},
        "requirement": {"genre": "悬疑", "request": "创作校园悬疑短剧"},
        "constraints": GenerationConstraints(),
        "writing_guidelines": [],
        "user_preferences": None,
    }


class SequenceLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def generate(self, messages, *, temperature=0.7):
        self.messages.append(messages)
        return next(self.responses)
