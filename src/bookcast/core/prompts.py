from __future__ import annotations


def build_language_output_requirement(language: str) -> str:
    normalized_language = str(language).strip() or "zh"
    return (
        f"由于当前要求的输出语言是{normalized_language}，"
        f"你的输出必须完全使用{normalized_language}，而且要求自然地道，避免翻译腔。"
    )


def build_chunk_card_prompt(chunk_text: str, section_title: str) -> str:
    if not section_title:
        return f"""
你是一个专业的图书内容分析助手。请阅读以下提供的文本切片，并提取核心信息，输出为一个 JSON 对象。

提取要求：
1. summary：用 100 到 200 字高度概括这段文字的核心内容。
2. key_points：提取 1 到 8 条关键事实、论证步骤或观点片段。
3. core_terms：提取 0 到 8 个该段落中出现的生僻概念、专有名词或核心术语，并结合原文语境给出简短解释。

原始文本：
{chunk_text}

输出格式：
请严格输出一个符合以下结构的 JSON，不要包含任何 Markdown 标记或额外说明文字：
{{
  "summary": "...",
  "key_points": ["...", "..."],
  "core_terms": [
    {{
      "term": "...",
      "explanation": "..."
    }}
  ]
}}
""".strip()

    return f"""
你是一个专业的图书内容分析助手。当前提供的文本切片（Chunk）属于章节：【{section_title}】。
请阅读该文本切片，提取与该章节主题相关的核心信息，并输出为一个 JSON 对象。

## 提取要求：
1. 强关联限制：提取内容时，**必须严格围绕章节【{section_title}】的主题进行**。对于与该章节主题无关的内容（例如前言引语、跨章节串联、排版残留等），请直接忽略，绝对不要提及。
2. summary：用 100 到 200 字高度概括这段文字中与【{section_title}】相关的核心内容。
3. key_points：提取 1 到 8 条属于该章节主题的关键事实、论证步骤或观点片段。
4. core_terms：提取 0 到 8 个该段落中出现的生僻概念、专有名词或核心术语，并结合原文语境给出简短解释。
5. 空值处理：如果判定整个切片的内容都与章节【{section_title}】完全无关，请将所有字段置空。

## 原始文本：
{chunk_text}

## 输出格式：
请严格输出一个符合以下结构的 JSON，不要包含任何 Markdown 标记或额外说明文字（如果整个切片无关，请输出空字符串和空数组）：
{{
  "summary": "...",
  "key_points": ["...", "..."],
  "core_terms": [
    {{
      "term": "...",
      "explanation": "..."
    }}
  ]
}}
""".strip()


def build_pseudo_section_prompt(all_chunk_cards_json: str) -> str:
    return f"""
你是一个资深的图书编辑和结构梳理专家。由于这本原书没有明确的章节目录，我为你提供了全书按顺序排列的所有文本切片卡片。
你的任务是：根据内容语义的连贯性和主题的转换，将这些连续的 Chunk 聚类划分成若干个逻辑上连贯的伪章节，并为每个伪章节生成一张总结卡片。

全书切片卡片集合：
{all_chunk_cards_json}

处理与提取要求：
1. 聚类划分：判断哪些连续的 chunk_id 在讨论同一个大主题，将它们划分为一个 pseudo-section。切分点应在语义明显转换或新主题开始的地方。
2. title：为每个划分出的伪章节提炼一个准确、凝练的标题。
3. source_chunk_ids：列出属于该伪章节的所有连续的 chunk_id。
4. summary：用 150 到 320 字连贯概括该伪章节的内容。
5. thesis_or_function：用 1 句话精准定义这部分内容在作者整体论述中的作用。
6. key_points：梳理 1 到 10 条核心论证过程。
7. core_terms：从这些 chunk 中挑选最重要的核心术语并解释。

输出格式：
请严格输出纯 JSON 对象，包含一个名为 pseudo_sections 的数组，数组中的每一个元素代表一个伪章节：
{{
  "pseudo_sections": [
    {{
      "source_chunk_ids": ["CH00_CK0001", "CH00_CK0002"],
      "title": "...",
      "summary": "...",
      "thesis_or_function": "...",
      "key_points": ["...", "..."],
      "core_terms": [
        {{
          "term": "...",
          "explanation": "..."
        }}
      ]
    }}
  ]
}}
""".strip()


def build_section_card_prompt(section_id: str, section_title: str, chunk_cards_json: str) -> str:
    return f"""
你是一个资深的图书编辑。请仔细阅读以下属于【{section_id} {section_title}】这一章的所有文本切片卡片，并综合生成该章节的总体总结卡片。

当前章节切片卡片集合：
{chunk_cards_json}

提取要求：
1. title：优化或保留原章节标题。
2. summary：用 150 到 320 字对本章节的总体内容进行连贯概括。
3. thesis_or_function：用 1 句话精准定义该章在全书中的作用。
4. key_points：梳理 1 到 10 条本章的核心论证过程或关键信息点。
5. core_terms：从提供的切片卡片中，挑选出对理解本章最重要的核心术语，并结合上下文优化解释。

输出格式：
请严格输出一个符合以下结构的纯 JSON 对象，不要包含任何 Markdown 代码块标记或额外说明文字：
{{
  "title": "...",
  "summary": "...",
  "thesis_or_function": "...",
  "key_points": [
    "...",
    "..."
  ],
  "core_terms": [
    {{
      "term": "...",
      "explanation": "..."
    }}
  ]
}}
""".strip()


def build_book_charter_prompt(
    book_title: str,
    all_section_cards_json: str,
    all_chunk_cards_json: str,
) -> str:
    return f"""
你是一个专业的播客制作人和图书拆解专家。请根据以下提供的全书章节级总结和切片级总结，生成这本书的总控理解文件。

书名：
{book_title}

全书章节总结卡片集合：
{all_section_cards_json}

全书切片总结卡片集合：
{all_chunk_cards_json}

任务要求：
1. book_summary：用300到800字浓缩全书内容，说明作者写作目的和主要结论。
2. global_theme：用一句话概括全书的核心立意或终极主旨。
3. core_argument_or_mainline：提炼全书最核心的论证主线，不超过五条。
4. global_terms：挑选贯穿全书、后续需要重点反复提及的极少数核心概念及解释。
5. planning_notes：站在播客编导视角，给出改编建议，例如哪些部分可压缩，哪些概念需要多举例，哪些逻辑链必须保全。

输出格式：
请严格输出一个 JSON 对象，不要包含 Markdown 代码块或额外说明：
{{
  "book_summary": "...",
  "global_theme": "...",
  "core_argument_or_mainline": [
    "...",
    "..."
  ],
  "global_terms": [
    {{
      "term": "...",
      "explanation": "..."
    }}
  ],
  "planning_notes": [
    "...",
    "..."
  ]
}}
""".strip()


def build_program_config_prompt(
    book_title: str,
    primary_language: str,
    mode: str,
    target_episode_minutes: int,
    allow_external_knowledge: bool,
    dialogue_mode: str,
    book_charter_json: str,
    user_customization_json: str,
) -> str:
    language_requirement = build_language_output_requirement(primary_language)
    return f"""
你是一个播客节目的总导演。你需要为一本即将被改编成播客的书籍设定节目框架。

用户输入与限制：
- 对话模式：{dialogue_mode}（如果是 single，该节目只有一位主持人；如果是 dual，则该节目有两位主持人）。
- 输出语种：{primary_language}
- 播客模式：{mode}（包含三种：faithful=忠于原文，要求生成的播客剧本不能增加或删减原文内容；concise=精炼总结，要求对原文进行精炼总结；deep_dive=深度讲解，要求对原文进行拓展或者深层次的讲解）。
- 每集目标时长：{target_episode_minutes} 分钟
- 是否允许补充外部知识：{allow_external_knowledge}

用户可选自定义偏好（仅在提供时优先遵循；未提供的字段请自行设计，不要输出"未提供"等占位说法）：
{user_customization_json}

书名：
{book_title}

图书总控信息：
{book_charter_json}

任务要求：
1. show_title：如果用户提供了节目名称偏好，请优先遵循并可做轻微润色；否则自行构思一个吸引人的播客栏目名。
2. positioning：如果用户提供了节目定位偏好，请优先遵循并补全为一句话的节目卖点和调性；否则自行设计。
3. target_audience：如果用户提供了目标听众偏好，请优先遵循并补全为清晰的听众画像；否则自行设计。
4. language_output_rules：设定语言策略和术语解释策略。
5. pace_style：描述节目节奏要求。
6. target_script_chars：根据时长预估单集剧本中文字数，按约每分钟三百字估算。
7. target_input_chars：预估单集需要消化的原文字符数。请务必结合播客模式来合理推算输入字数与目标剧本字数的比例关系。根据播客模式和target_script_chars进行估算，faithful=忠于原文，要求生成的播客剧本不能增加或删减原文内容，即target_input_chars应当与target_script_chars大致相同；concise=精炼总结，要求对原文进行精炼总结，输入的原文一般大于生成的剧本长度，即target_input_chars应该大于target_script_chars；deep_dive=深度讲解，要求对原文进行拓展或者深层次的讲解，输入的原文一般小于生成的剧本长度，即target_input_chars应该小于target_script_chars。
8. tone_guardrails：制定语气护栏。
9. content_guardrails：制定内容护栏。

输出格式：
{language_requirement}
请严格输出一个 JSON 对象，不要包含 Markdown 代码块或额外说明：
{{
  "show_title": "...",
  "positioning": "...",
  "target_audience": "...",
  "language_output_rules": {{
    "script_language": "...",
    "term_policy": "..."
  }},
  "pace_style": "...",
  "target_script_chars": ...,
  "target_input_chars": ...,
  "tone_guardrails": ["...", "..."],
  "content_guardrails": ["...", "..."]
}}
""".strip()


def build_speaker_bible_prompt(
    dialogue_mode: str,
    primary_language: str,
    show_title: str,
    positioning: str,
    target_audience: str,
    user_customization_json: str,
) -> str:
    language_requirement = build_language_output_requirement(primary_language)
    return f"""
你是一个专业的声优导演和播客编剧。请根据当前的节目级设定，为播客创建具体的主持人人设圣经。

节目设定与用户偏好：
对话模式：{dialogue_mode}
节目名称：{show_title}
节目定位：{positioning}
目标听众：{target_audience}

用户可选自定义偏好（仅在提供时优先遵循；未提供的字段请自行设计，不要输出"未提供"等占位说法）：
{user_customization_json}

任务要求：
生成详细的角色设定，以确保后续剧本生成中角色的声音一致性。
如果是双人模式（dual），必须设定两位主持人（S1 主讲推进，S2 听众代理与追问），明确分配他们的词汇偏好、句长倾向、转场方式、禁止的口头禅、节目特有金句等。并制定两人的互动原则。
S1 和 S2 的目标发言比例（target_share_percent）总和必须为 100。
如果是单人模式（single），只需设定一位主持人（S1 主讲），明确分配其词汇偏好、句长倾向、转场方式、禁止的口头禅等。S1 的目标发言比例（target_share_percent）必须为 100。
如果用户提供了主持人名称偏好，请优先使用这些名字；如果用户提供了主持人性格特点，请将其吸收到 persona_summary、tone、词汇与转场风格中，而不是只机械复述一句提示词。

请确保输出的 JSON 严格满足以下每个字段的定义和要求：
- fixed_opening: 如果用户提供了 fixed_opening 偏好，请优先遵循其核心表达，并结合节目名称、主持人名字和对话模式做必要润色；如果没有提供，再自行设计。请务必根据对话模式是单人还是双人来安排打招呼的人数和话术。使用 [S1] / [S2] 来标记说话人。
  一个单人的例子："[S1]欢迎来到{show_title}，今天我们继续往下聊。"
  一个双人的例子："[S1]欢迎来到{show_title}，今天是xx年xx月xx日。我是小苔藓。[S2]听众们好，我是小皮球。"
- speakers -> display_name: 为主持人起的名字；如果用户提供了名称偏好，请优先遵循。
- speakers -> role: 该角色在节目中的功能定位。
- speakers -> persona_summary: 角色的性格速写；如果用户提供了性格特点，请优先吸收并展开。
- speakers -> tone: 具体的声音情绪和语气指导。
- speakers -> vocabulary_preferences: 专属的高频词汇或句式。
- speakers -> sentence_length_tendency: 句长偏好（long, short, mixed）。
- speakers -> transition_patterns: 该角色常用的转场或承上启下的话术模板。
- speakers -> banned_catchphrases: 绝对不能使用的口语词。
- speakers -> allowed_show_phrases: 节目特有的标志性口头语。
- speakers -> target_share_percent: 两人发言字数的大致比例预期，单人时 S1 为 100。
- interaction_rules: 双人模式专属，两人交谈时的互动原则、打断机制。单人时为空。
- consistency_rules: 确保跨集人设不崩塌的原则要求。
- forbidden_behaviors: 严禁角色做出的破坏人设或节目连贯性的行为。

输出格式：
{language_requirement}
请严格输出一个纯 JSON 对象，不要包含任何 Markdown 代码块标记：
{{
  "fixed_opening": "...",
  "speakers": [
    {{
      "display_name": "...",
      "role": "...",
      "persona_summary": "...",
      "tone": "...",
      "vocabulary_preferences": ["...", "..."],
      "sentence_length_tendency": "...",
      "transition_patterns": ["...", "..."],
      "banned_catchphrases": ["...", "..."],
      "allowed_show_phrases": ["...", "..."],
      "target_share_percent": ...
    }},
    {{
      "display_name": "...",
      "role": "...",
      "persona_summary": "...",
      "tone": "...",
      "vocabulary_preferences": ["...", "..."],
      "sentence_length_tendency": "...",
      "transition_patterns": ["...", "..."],
      "banned_catchphrases": ["...", "..."],
      "allowed_show_phrases": ["...", "..."],
      "target_share_percent": ...
    }}
  ],
  "interaction_rules": ["...", "..."],
  "consistency_rules": ["...", "..."],
  "forbidden_behaviors": ["...", "..."]
}}
""".strip()


def build_episode_plan_prompt(
    target_script_chars: int,
    target_input_chars: int,
    mode: str,
    primary_language: str,
    positioning: str,
    target_audience: str,
    core_argument_or_mainline_json: str,
    all_cards_summary_json: str,
) -> str:
    language_requirement = build_language_output_requirement(primary_language)
    return f"""
你是一个资深的播客统筹编导。请根据本书的总体规划和所有章节、切片内容，制定出整季播客的分集计划。

节目限制与总控：
- 目标单集字数：{target_script_chars}。这是生成的每集播客剧本的目标字数要求。
- 目标输入字数：{target_input_chars}。这是每集需要参考的原文字数要求。
- 也就是说，你规划的每一集，其 covers 中所有 chunk 的原文总字数应当控制在 {target_input_chars} 左右；然后再根据节目模式 {mode} 的要求，生成目标字数在 {target_script_chars} 左右的讲稿。
- 节目模式：{mode}
- 节目定位：{positioning}
- 目标听众：{target_audience}
- 全书核心主线：{core_argument_or_mainline_json}

全书章节与切片信息摘要：
{all_cards_summary_json}

任务要求：
请科学地将这本书拆分为多个单集播客，并计算 total_episode_count。对于每一集，你必须明确：
1. title：单集标题。
2. section_ids / covers：本集主讲的章节 ID 和切片 ID 列表。请结合每个 chunk 的 `length_stats.char_count` 估算 covers 的总原文字数，并尽量控制在 target_input_chars 左右。
3. neighbor_context：用于衔接的相邻切片 ID。
4. must_cover：必须讲清的重点。
5. can_skip：允许省略的次要背景。
6. forbidden_to_introduce：严禁提前剧透的概念。
7. hook：开场钩子。
8. recap_focus：需要回顾的上一集关键信息，第一集可为空。
9. teaser_goal：结尾悬念，引向下一集。
10. tone_target：本集的局部语气。

输出格式：
{language_requirement}
请严格输出一个符合以下结构的纯 JSON 对象，不要包含任何 Markdown 代码块标记：
{{
  "total_episode_count": ...,
  "episodes": [
    {{
      "title": "...",
      "section_ids": ["CH01"],
      "covers": ["CH01_CK0001", "CH01_CK0002"],
      "neighbor_context": ["CH01_CK0003"],
      "must_cover": ["...", "..."],
      "can_skip": ["...", "..."],
      "forbidden_to_introduce": ["...", "..."],
      "hook": "...",
      "recap_focus": "...",
      "teaser_goal": "...",
      "tone_target": "..."
    }}
  ]
}}
""".strip()


def build_episode_script_prompt(
    speaker_bible_json: str,
    program_config_json: str,
    episode_plan_json: str,
    recent_episode_cards_json: str,
    series_memory_summary_json: str,
    last_episode_tail_excerpt: str,
    raw_text_chunks_json: str,
    mode: str,
    target_script_chars: int,
    primary_language: str,
) -> str:
    language_requirement = build_language_output_requirement(primary_language)
    return f"""
你是一位顶尖的播客文字脚本撰稿人。请根据提供的 Source Pack，生成一段绝佳的播客对话脚本。

动态规则注入：
- 角色设定与分工：{speaker_bible_json}
- 节目定位与护栏：{program_config_json}
- 本集计划：{episode_plan_json}
- 最近几集回顾：{recent_episode_cards_json}
- 历史记忆与上下文：{series_memory_summary_json}
- 上一集结尾：{last_episode_tail_excerpt}

核心原始资料：
{raw_text_chunks_json}

写作具体要求：
一、语言风格
- 严格遵循 speaker_bible 中定义的词汇偏好、口头禅黑白名单及句长倾向。
- 使用自然、随意、轻松的日常表达，优先采用简单词汇，将书面转换为口语。
- 适度加入符合人设的语气词。

二、句式结构
- 使用松散、自然的句式，允许存在口语特征如重复、停顿、语气词等；
- 鼓励使用叠词（如"特别特别"、"慢慢来"）和填充词（如"这个"、"其实"、"然后"、"就是"、"呃"等）；
- 可适度插入模糊表达、略带情绪的语调，增强亲和力。

三、对话结构
- 如果 speaker_bible 中只有一位主持人，只在开头使用 [S1] 标记说话人轮次，其他地方不要出现说话人标记。
- 如果 speaker_bible 中有两位主持人，使用 [S1] 和 [S2] 标记说话人轮次，且标记中间不换行。
- 每当一方讲话时，另一方可以适当插入自然、简短的反馈或承接语（如"嗯。""对。""是的。""确实。""原来是这样。"等），展现倾听状态；
    - 对话应有开头引入、核心讨论与自然结尾，语气上有节奏起伏，避免平铺直叙；
- **特别强调听话方的积极反馈：当一位说话人正在讲述或解释某个观点时，另一位说话人应频繁地插入简短的承接或反馈词语（例如：「嗯。」、「是。」、「对。」、「哦。」、「是的。」、「哦，原来是这样。」、「明白。」、「没错。」、「有道理。」、「确实」），以表明其正在积极倾听、理解和互动。这些反馈应自然地穿插在说话者语句的间歇或段落转换处，而不是生硬地打断。例如："[S2]我本人其实不太相信星座诶，[S1]嗯。[S2]在一开始的时候，我就跟大部分不相信星座的一样，觉得，呃，你总能把人就分成十二种，[S1]是的。[S2]然后呢就它讲的就是对的。",这种反馈要尽可能多，不要吝啬。**
- 根据 episode_plan 的 hook 开场，围绕 must_cover 展开，以 teaser_goal 结尾。

四、标点与格式
- 仅使用中文标点：逗号、句号、问号；
    - 禁止使用叹号。禁止使用省略号（'...'）、括号、引号（包括''""'"）或波折号等特殊符号；
    - 所有数字转换为中文表达，如"1000000"修改为"一百万"；
    - 请根据上下文，智慧地判断数字的读音，所有带数字的英文缩写要意译，如"a2b"输出为"a到b"、"gpt-4o"输出为"GPT四O"、"3:4"输出为"3比4"，"2021"如果表达年份，应当转换为"二零二一"，但如果表示数字，应当转换为"两千零二十一"。请保证不要简单转换为中文数字，而是根据上下文，将其翻译成合适的中文。


五、内容要求
- 不得照搬原始资料的书面表达，原始资料信息需完整提及。
- 涉及抽象技术点或 episode_plan 提示的复杂概念，需使用生动的比喻。
- 不要在对话内输出我是谁等生硬介绍。

输出要求：
{language_requirement}

请直接输出剧本结果，不要包含任何额外信息或 Markdown 代码块标记，纯文本输出。
再次强调：最终生成的整集剧本文字长度应控制在 {target_script_chars} 左右，不要明显过短，也不要明显超出太多。
你应当遵循的播客模式是 {mode}。
- faithful 表示忠于原文，要求生成的播客剧本不能增加或删减原文内容。
- concise 表示精炼总结，要求对原文进行精炼总结。
- deep_dive 表示深度讲解，要求对原文进行适度拓展，或者进行更深层次的讲解。
""".strip()


def build_episode_memory_prompt(
    episode_plan_json: str,
    generated_script: str,
    old_memory_json: str,
) -> str:
    return f"""
你是一个播客剧本审查与场记专员。刚才我们生成了最新一集的播客剧本。请仔细阅读本集原始计划、实际剧本内容和旧版全局记忆，输出单集记录卡并更新全季全局记忆。

本集原始计划：
{episode_plan_json}

本集实际生成的剧本：
{generated_script}

剧本生成前的旧版全局记忆：
{old_memory_json}

任务要求：
你需要返回一个包含两个对象的 JSON 结构：episode_card 和 updated_series_memory。

对于 episode_card：
1. title：修正或确定该集的最终标题。
2. summary：用 120 到 240 字客观总结这集剧本实际讲了什么。
3. covered_sections / covered_chunks：记录实际消耗的章节和切片 ID。
4. introduced_terms：提取本集中向听众详细解释过的新术语。
5. callbacks / resolved_loops / open_loops：提取剧本中的呼应与悬念。
6. tone_notes：对本集实际表现出的风格做标注评估。
7. do_not_repeat_next_episode：给下一集编剧的强制备忘录。

对于 updated_series_memory：
1. cumulative_summary：扩写到目前为止的整体前情提要。
2. section_coverage_status：更新章节消耗状态。
3. introduced_terms_global：维护全局已解释术语及其首次出现的 episode_id。
4. callbacks_global / open_loops_global / resolved_loops_global：更新全局呼应与悬念池。
5. repetition_watchlist：更新下一集不应重复的内容。

输出格式：
请严格输出一个符合以下结构的纯 JSON 对象，不要包含任何 Markdown 代码块标记：
{{
  "episode_card": {{
    "title": "...",
    "summary": "...",
    "covered_sections": ["..."],
    "covered_chunks": ["..."],
    "introduced_terms": ["...", "..."],
    "callbacks": ["..."],
    "resolved_loops": ["..."],
    "open_loops": ["..."],
    "tone_notes": "...",
    "do_not_repeat_next_episode": ["..."]
  }},
  "updated_series_memory": {{
    "cumulative_summary": "...",
    "section_coverage_status": [
      {{
        "section_id": "CH01",
        "status": "done",
        "episode_ids": ["E001"]
      }}
    ],
    "introduced_terms_global": [
      {{
        "term": "...",
        "first_episode_id": "..."
      }}
    ],
    "callbacks_global": ["..."],
    "open_loops_global": ["..."],
    "resolved_loops_global": ["..."],
    "repetition_watchlist": ["..."]
  }}
}}
""".strip()
