from pydantic import BaseModel


class Character(BaseModel):
    name: str
    role: str
    personality: str
    goal: str


class Dialogue(BaseModel):
    character: str
    line: str
    emotion: str | None = None


class Scene(BaseModel):
    scene_id: int
    location: str
    time: str
    purpose: str
    conflict: str
    visual_action: str
    dialogues: list[Dialogue]
    estimated_seconds: int
    ending_hook: str | None = None
    shooting_note: str | None = None


class Screenplay(BaseModel):
    title: str
    genre: str
    logline: str
    theme: str
    opening_hook: str
    main_conflict: str
    reversal: str
    emotional_payoff: str
    characters: list[Character]
    scenes: list[Scene]
    ending: str


class GenerationConstraints(BaseModel):
    target_duration_seconds: int = 240
    min_duration_seconds: int = 180
    max_duration_seconds: int = 300
    target_scene_count: int = 4
    min_scene_count: int = 3
    max_scene_count: int = 6
    max_json_repair_attempts: int = 2
    max_content_revisions: int = 2
    pass_score: float = 8.0


def validate_screenplay(
    screenplay: Screenplay,
    constraints: GenerationConstraints,
) -> list[str]:
    errors: list[str] = []
    duration = sum(scene.estimated_seconds for scene in screenplay.scenes)
    if not constraints.min_duration_seconds <= duration <= constraints.max_duration_seconds:
        errors.append("总时长不符合约束")

    scene_count = len(screenplay.scenes)
    if not constraints.min_scene_count <= scene_count <= constraints.max_scene_count:
        errors.append("场景数量不符合约束")

    character_names = {character.name for character in screenplay.characters}
    unknown_characters = {
        dialogue.character
        for scene in screenplay.scenes
        for dialogue in scene.dialogues
        if dialogue.character not in character_names
    }
    if unknown_characters:
        errors.append(f"对白引用了未知角色：{', '.join(sorted(unknown_characters))}")

    scene_ids = [scene.scene_id for scene in screenplay.scenes]
    if scene_ids != list(range(1, len(scene_ids) + 1)):
        errors.append("scene_id 必须唯一且从 1 连续递增")

    if not screenplay.opening_hook.strip() or not any(
        scene.ending_hook and scene.ending_hook.strip() for scene in screenplay.scenes
    ):
        errors.append("剧本必须包含开头钩子和至少一个场景结尾钩子")

    if not screenplay.reversal.strip():
        errors.append("剧本必须包含明确反转")
    return errors
