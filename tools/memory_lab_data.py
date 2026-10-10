"""Deterministic fictional data and independently labelled retrieval questions.

Nothing here is an actual fact about the owner of this workspace. Categories
remain the application's four categories; domains are only fixture metadata.
"""
from __future__ import annotations

import json
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone

SOURCE = 'memory-lab-synthetic-v1'


def fact(key, content, category='note', domain='daily', pinned=False):
    return dict(key=key, content=content, category=category, domain=domain, pinned=int(pinned))


ANCHORS = [
    fact('basic.name', '这是虚构测试人物小林，称呼用小林。', 'profile', pinned=True),
    fact('basic.language', '默认用简体中文回答，技术名词可以保留英文。', 'preference', pinned=True),
    fact('basic.style', '回答先给结论，再给必要的解释；不要长篇铺垫。', 'preference', pinned=True),
    fact('basic.units', '默认使用公制单位和摄氏度。', 'preference', pinned=True),
    fact('basic.city', '小林当前居住在悉尼，日程使用 Australia/Sydney 时区。', 'profile', pinned=True),
    fact('basic.platform', '小林的开发电脑运行 Windows 11，终端使用 PowerShell。', 'profile', 'development', True),
    fact('dev.python', '个人网页采集项目用 Python 和 aiohttp 实现异步爬虫，并发上限为八个请求。', 'project', 'development'),
    fact('food.coffee', '小林下午三点以后不喝咖啡，避免影响晚上睡眠。', 'preference', 'food'),
    fact('work.aurora', 'Aurora 项目的后端是 FastAPI，数据库是 PostgreSQL，缓存使用 Redis。', 'project', 'work'),
    fact('travel.kyoto', '京都旅行计划安排在十一月，优先住四条附近安静的小旅馆。', 'project', 'travel'),
    fact('dev.rime', 'Rime 输入法插件用 C++ 开发，只在 Windows 上构建，暂时不支持 macOS。', 'project', 'development'),
    fact('dev.node', 'Node.js 工具项目使用 pnpm 管理依赖，测试用 Vitest。', 'project', 'development'),
    fact('dev.async', '异步数据库练习里使用 asyncio 和 asyncpg，不使用同步 psycopg。', 'project', 'development'),
    fact('study.exam', 'ELEC5619 的复习顺序是会话安全、CSRF、数据库事务和部署。', 'project', 'study'),
    fact('study.english', '小林练习英语演讲时需要短句和常用词，避免复杂从句。', 'preference', 'study'),
    fact('study.math', '线性代数复习先学特征值和特征向量，再学奇异值分解 SVD。', 'project', 'study'),
    fact('fitness.knee', '跑步后左膝容易不舒服，训练计划优先游泳和低冲击运动。', 'preference', 'fitness'),
    fact('fitness.swim', '每周二和周四下班后游泳，每次约四十分钟。', 'preference', 'fitness'),
    fact('fitness.bike', '通勤自行车是 Trek FX 3，轮胎尺寸为七百乘三十二。', 'profile', 'fitness'),
    fact('food.milk', '点拿铁时选燕麦奶，因为喝普通牛奶容易肠胃不适。', 'preference', 'food'),
    fact('food.spicy', '吃饭能接受微辣，不能接受重辣；面条不要加香菜。', 'preference', 'food'),
    fact('food.apple', '买水果时喜欢青苹果，通常买 Granny Smith，不喜欢太甜的红苹果。', 'preference', 'food'),
    fact('travel.hotel', '旅行酒店优先安静、独立卫浴和步行可达公共交通，不优先选夜生活街区。', 'preference', 'travel'),
    fact('travel.passport', '虚构旅行清单要求出发前检查护照有效期和旅行保险，不保存证件号码。', 'note', 'travel'),
    fact('travel.nz', '新西兰南岛路线计划包含蒂卡波、库克山和皇后镇，安排十天。', 'project', 'travel'),
    fact('home.bulb', '书房灯泡是 E27 灯口，喜欢暖白光，色温约三千开尔文。', 'preference', 'home'),
    fact('home.router', '家用路由器放在客厅书架中层，有线网络连接书房电脑。', 'note', 'home'),
    fact('home.plant', '阳台的龟背竹放在散射光处，表土干燥后再浇水。', 'note', 'garden'),
    fact('pet.cat', '小林养的猫叫芝麻，是三岁的狸花猫；宠物不是个人的称呼。', 'profile', 'pets'),
    fact('pet.food', '芝麻的猫粮选鸡肉口味，换粮要逐步混合，避免突然更换。', 'preference', 'pets'),
    fact('hobby.camera', '摄影相机是 Fujifilm X-S20，旅行通常带二十三毫米定焦镜头。', 'profile', 'photography'),
    fact('hobby.photo', '拍照偏好自然光和低饱和色调，少用强烈 HDR。', 'preference', 'photography'),
    fact('reading.book', '最近在读 Designing Data-Intensive Applications，重点关注存储和复制章节。', 'project', 'reading'),
    fact('reading.fiction', '小说偏好科幻和推理，喜欢章节较短的作品。', 'preference', 'reading'),
    fact('finance.budget', '虚构月度预算用电子表格记录固定支出、可变支出和储蓄；金额只是测试用例。', 'project', 'finance'),
    fact('finance.savings', '旅行储蓄目标在测试表中单独记录，不能和账户余额混为一项。', 'note', 'finance'),
    fact('device.apple', '工作手机是 Apple iPhone 15；手机品牌和买水果的苹果偏好是两个主题。', 'profile', 'devices'),
    fact('device.mouse', '书房电脑鼠标是 Logitech MX Master 3S，侧键绑定切换桌面。', 'profile', 'devices'),
    fact('garden.mouse', '阳台种植日志记录过老鼠咬坏花盆边缘，需要检查防鼠网。', 'note', 'garden'),
    fact('work.meeting', 'Aurora 项目每周三上午十点开例会，会议纪要记录结论、负责人和截止日期。', 'project', 'work'),
    fact('work.slides', '项目展示材料希望用简单英文，突出真实完成的工作，不夸大贡献。', 'preference', 'work'),
    fact('music.piano', '钢琴练习使用节拍器，从每分钟六十拍开始，再逐步加速。', 'project', 'music'),
    fact('music.listen', '专注工作时喜欢无歌词的钢琴音乐，避免人声打断阅读。', 'preference', 'music'),
    fact('game.city', '城市建造游戏存档名为云山，当前在设计铁路与居民区之间的绿化隔离带。', 'project', 'gaming'),
    fact('dev.city', 'Three.js 城市场景使用 Vite 构建，建筑高度需要考虑地形，浏览器画面验收单独进行。', 'project', 'development'),
    fact('note.old.city', '历史搬家笔记：去年住在墨尔本；后来已搬到悉尼，这不是当前居住地。', 'note', 'daily'),
    fact('note.old.coffee', '历史饮食日志：过去下午喝咖啡；最近已经改为下午三点以后不喝。', 'note', 'food'),
    fact('note.quoted', '写作练习的虚构台词是“我最喜欢重辣”，台词不是小林的饮食偏好。', 'note', 'reading'),
]

# Diverse distractors, rather than hundreds of copies of one keyword template.
DOMAINS = {
    'development': ['Git 分支在发布前检查差异', 'Docker 开发容器只映射本机端口', 'SQLite 备份先检查 WAL 文件', 'CSS 布局关注小屏幕体验', 'Rust 练习借用检查器', 'TypeScript 配置启用严格模式', 'Linux 命令练习文件权限', '日志不打印访问凭据'],
    'study': ['统计学复习置信区间', '电路课整理戴维南等效', '英语听力每天做短练习', '离散数学练习图论', '实验报告先描述观测结果', '学习笔记按概念整理', '考试前整理错题', '小组作业注明参考资料'],
    'work': ['设计评审保留决策依据', '文档评审关注接口兼容', '例会只讨论阻塞问题', '看板记录真实进度', '客户演示先准备测试环境', '代码评审先看行为变化', '交接材料包含运行步骤', '季度计划保留缓冲时间'],
    'food': ['早餐喜欢无糖酸奶和坚果', '晚餐常做清蒸鱼', '备餐盒选择可微波材质', '周末烤面包降低糖量', '红茶通常不加糖', '蔬菜采购按季节选择', '厨房调料按开封日期整理', '意面偏好番茄蘑菇酱'],
    'fitness': ['散步路线沿河岸步道', '健身后做肩背拉伸', '午休做短时间活动', '运动水壶用不锈钢材质', '羽毛球练习正手高远球', '运动记录包括睡眠时长', '力量训练先练基本动作', '徒步鞋注重防滑'],
    'travel': ['短途旅行带轻量背包', '机场接驳预留交通时间', '步行路线包含休息点', '博物馆参观避开人流高峰', '行李清单分衣物和电子设备', '旅行照片按目的地归档', '火车出行优先靠窗座位', '露营装备检查帐篷防雨'],
    'home': ['洗衣用品放在洗衣房上层', '桌面收纳使用透明盒', '客厅窗帘定期清洗', '书柜按书籍高度分层', '扫地机地图标记地毯', '空气净化器滤芯按日期记录', '卧室床品选择棉质', '工具箱按用途分格'],
    'pets': ['宠物外出包放在玄关', '玩具定期检查破损', '饮水碗每天清洁', '宠物照片按月份归档', '猫抓板放在沙发旁', '护理日志记录梳毛日期', '旅行前安排宠物照看', '宠物就诊记录保留机构名称'],
    'photography': ['照片导出保留原始文件', '相机包选择轻量肩带', '雨天拍摄准备防水罩', '构图练习使用三分法', '街头照片避免过度锐化', '风景拍摄关注日落时刻', '镜头布定期更换', '作品集按主题编排'],
    'reading': ['读书笔记记录问题和观点', '纸质书偏好轻便装帧', '周末阅读安排安静时段', '非虚构书先浏览目录', '书摘注明页码', '电子书字体选择无衬线', '借阅书籍记录归还日期', '故事练习关注人物动机'],
    'finance': ['测试账本区分收入与转账', '虚构票据按月份整理', '测试表检查公式引用', '订阅清单按续费日排序', '预算练习区分必要和可选支出', '测试图表避免重复统计', '虚构报销单保留类别', '示例数据不包含真实账号'],
    'devices': ['键盘使用静音轴', '显示器支架调整观看高度', '充电线按接口分类', '耳机偏好舒适佩戴', '旧硬盘先备份再处理', '打印机纸张放在干燥处', '移动电源标注容量', '摄像头设置只保留必要权限'],
    'garden': ['盆栽记录换盆日期', '肥料标签注明适用植物', '剪枝工具使用后擦干', '香草种子按季节播种', '雨天减少浇水次数', '土壤配比记录珍珠岩比例', '花盆托盘定期排水', '育苗日志记录发芽时间'],
    'music': ['乐谱按练习难度整理', '录音文件保留节奏标记', '耳训先练纯五度', '吉他练习慢速换和弦', '练习日志记录容易出错的小节', '音乐设备放在干燥处', '演奏前做手指热身', '播放列表按使用场景分类'],
    'gaming': ['解谜游戏记录未完成线索', '策略游戏保存不同阶段存档', '控制器按键使用默认布局', '游戏截图按作品分类', '赛车练习关注弯道刹车点', '沙盒地图标记资源位置', '合作任务记录队友分工', '游戏时间安排在周末'],
}


def dataset(size):
    if size < 1:
        raise ValueError('size must be positive')
    facts = list(ANCHORS[:size])
    domains = list(DOMAINS)
    i = 0
    while len(facts) < size:
        domain = domains[i % len(domains)]
        round_no = i // len(domains)
        theme = DOMAINS[domain][round_no % len(DOMAINS[domain])]
        category = ('project', 'note', 'preference', 'note', 'project', 'profile')[round_no % 6]
        # Archive/project identifiers make records independently editable. A
        # different place, date, qualifier and amount add lexical distractors.
        location = ('家里', '图书馆', '办公室', '工作室', '学校', '社区')[round_no % 6]
        content = f'虚构测试人物的{location}档案 {round_no + 1:03d}：{theme}。适用范围是该档案，复查间隔为{7 + round_no % 30}天。'
        facts.append(fact(f'{domain}.archive.{round_no + 1:03d}', content, category, domain))
        i += 1
    # Fixed times and UUIDs make before/after comparisons reproducible.
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i, f in enumerate(facts):
        stamp = (epoch + timedelta(minutes=i)).isoformat(timespec='milliseconds')
        rows.append(dict(id=uuid.uuid5(uuid.NAMESPACE_URL, 'mori-memory-lab/' + f['key']).hex,
            **f, source=SOURCE, evidence='完全虚构的压力测试数据；不是用户的真实记忆。',
            created_at=stamp, updated_at=stamp))
    return rows


def case(name, query, keys=(), kind='keyword'):
    return dict(name=name, query=query, expected_keys=list(keys), kind=kind)


CASES = [
    case('异步爬虫', 'Python aiohttp 异步爬虫并发上限', ['dev.python']),
    case('项目后端', 'Aurora 后端数据库技术栈', ['work.aurora']),
    case('输入法平台', 'Rime C++ 支持什么系统', ['dev.rime']),
    case('Node 测试', 'Node pnpm Vitest', ['dev.node']),
    case('异步数据库', 'asyncpg asyncio 数据库练习', ['dev.async']),
    case('课程复习', 'ELEC5619 CSRF 复习顺序', ['study.exam']),
    case('英语演讲', '英语演讲用短句', ['study.english']),
    case('数学复习', '线性代数 SVD 特征值', ['study.math']),
    case('膝盖运动', '左膝 跑步 低冲击', ['fitness.knee']),
    case('游泳日程', '每周 游泳 时间', ['fitness.swim']),
    case('自行车型号', '通勤自行车 Trek', ['fitness.bike']),
    case('咖啡时间', '下午 咖啡 睡眠', ['food.coffee']),
    case('奶制品', '拿铁 燕麦奶 肠胃', ['food.milk']),
    case('饮食口味', '吃面条 辣 香菜 偏好', ['food.spicy']),
    case('苹果水果', '水果 青苹果 Granny Smith', ['food.apple']),
    case('京都行程', '京都 十一月 四条 住宿', ['travel.kyoto']),
    case('酒店偏好', '旅行 酒店 安静 卫浴', ['travel.hotel']),
    case('南岛路线', '新西兰 南岛 皇后镇', ['travel.nz']),
    case('灯泡规格', '书房 灯泡 E27 色温', ['home.bulb']),
    case('路由器位置', '家用路由器 客厅 位置', ['home.router']),
    case('植物浇水', '龟背竹 浇水 散射光', ['home.plant']),
    case('猫的名字', '猫 叫 芝麻 几岁', ['pet.cat']),
    case('猫粮偏好', '芝麻 猫粮 口味', ['pet.food']),
    case('相机镜头', 'Fujifilm X-S20 旅行 镜头', ['hobby.camera']),
    case('照片风格', '自然光 HDR 饱和 色调', ['hobby.photo']),
    case('技术书', 'Designing Data-Intensive Applications 章节', ['reading.book']),
    case('储蓄与余额', '旅行 储蓄 账户余额', ['finance.savings']),
    case('苹果手机', 'Apple iPhone 工作手机', ['device.apple']),
    case('电脑鼠标', '电脑 鼠标 Logitech 侧键', ['device.mouse']),
    case('阳台老鼠', '阳台 老鼠 花盆 防鼠网', ['garden.mouse']),
    case('会议安排', 'Aurora 例会 周三 时间', ['work.meeting']),
    case('钢琴练习', '钢琴 节拍器 六十拍', ['music.piano']),
    case('城市场景', 'Three.js Vite 建筑 地形', ['dev.city']),
    case('跨领域请求', '京都 旅行 住宿 拍照 Fujifilm 镜头', ['travel.kyoto', 'hobby.camera'], 'multi_domain'),
    case('当前居住地', '现在住在哪个城市', ['basic.city'], 'temporal'),
    case('隐含咖啡问法', 'afternoon stimulant cutoff', ['food.coffee'], 'paraphrase'),
    case('自行车同义改写', '两轮代步工具的品牌型号', ['fitness.bike'], 'paraphrase'),
    case('牛奶同义改写', '乳糖不耐受的时候饮品怎么选', ['food.milk'], 'paraphrase'),
    case('无记忆的项目', 'helium fusion reactor', [], 'negative'),
    case('中文无关问题', '量子纠缠实验设备', [], 'negative'),
    case('普通问候', '你好呀', [], 'negative'),
]


def applicable_cases(rows):
    keys = {r['key'] for r in rows}
    return [c for c in CASES if not c['expected_keys'] or set(c['expected_keys']) <= keys]


def manifest(rows):
    return dict(synthetic=True, seed=SOURCE, count=len(rows),
                domains=dict(Counter(r['domain'] for r in rows)),
                categories=dict(Counter(r['category'] for r in rows)),
                pinned=sum(r['pinned'] for r in rows),
                cases=applicable_cases(rows))


def seed(store, rows):
    """Never replace existing data, including edits made while trying the lab."""
    if store.memories():
        return False
    with store.connect() as db:
        db.executemany('''INSERT INTO memories VALUES
            (:id,:key,:content,:category,:pinned,:source,:evidence,:created_at,:updated_at)''', rows)
    return True
