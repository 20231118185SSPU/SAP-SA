from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

doc = Document()

# ===== 样式设置 =====
style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(12)
style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

# ===== 标题 =====
title = doc.add_heading('', level=0)
run = title.add_run('GaN HEMT 陷阱态在多物理场作用下的动态演化研究综述')
run.font.size = Pt(22)
run.font.name = 'Times New Roman'
run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

# 作者
author = doc.add_paragraph()
author.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = author.add_run('小天 | 上海第二工业大学 新能源科学与工程专业')
r.font.size = Pt(11)

# ===== 摘要 =====
h = doc.add_heading('摘要', level=1)
doc.add_paragraph(
    'GaN HEMT 因其高电流密度、高击穿电压、高功率密度和高频工作特性，被视为下一代功率器件的有力候选。'
    '然而，陷阱态（trap state）导致的稳定性与可靠性问题严重制约了 GaN HEMT 的发展。'
    '本研究聚焦于陷阱态在多物理场（电-热-应力）耦合下的动态演化机制，'
    '旨在通过长期应力测试 + 陷阱态表征 + TCAD 仿真，'
    '建立宏观失效-微观陷阱态演化的关联模型，为 GaN HEMT 可靠性评估提供理论支撑。'
)

# ===== 第1章 =====
doc.add_heading('第1章 研究背景与意义', level=1)
doc.add_paragraph(
    'GaN HEMT 因其高电流密度、高击穿电压、高功率密度和高频工作特性，'
    '被视为功率器件的 promising candidate。然而，陷阱态（trap state）导致的稳定性与可靠性问题'
    '严重制约了 GaN HEMT 的发展 [1]。'
)
doc.add_paragraph(
    '陷阱态引起的电流坍塌（current collapse）、导通电阻动态增加（dynamic R_ON increase）、'
    '阈值电压漂移（V_th shift）等现象，本质上是陷阱态在多物理场（电-热-应力）作用下的'
    '充放电动态演化过程 [2][4][8]。'
)
doc.add_paragraph(
    '本研究聚焦于陷阱态在多物理场耦合下的动态演化机制，旨在通过长期应力测试 + 陷阱态表征 + '
    'TCAD 仿真，建立宏观失效-微观陷阱态演化的关联模型，为 GaN HEMT 可靠性评估提供理论支撑。'
)

# ===== 第2章 =====
doc.add_heading('第2章 研究脉络与技术演进', level=1)

table = doc.add_table(rows=8, cols=3, style='Table Grid')
headers = ['时间', '里程碑', '贡献']
for j, h in enumerate(headers):
    table.cell(0, j).text = h
data = [
    ('2014', 'EC-0.57 eV 陷阱态定位', '确认 GaN buffer 中 EC-0.57 eV 陷阱是导致 drain-lag、RF 功率退化、电流坍塌的关键缺陷 [7]'),
    ('2015', 'Fe/C 掺杂陷阱态研究', 'Fe 掺杂引入 Ea=0.6 eV 陷阱导致电流坍塌；C 掺杂引入 Ev+0.84 eV 陷阱导致动态 R_ON 增加 [5]'),
    ('2015', '硅基底 MIS-HEMT 陷阱态机制', '识别三种主导陷阱态机制：OFF 态栅-漏 access 区电荷陷阱、SEMI-ON 态热电子注入、正栅偏压下栅介质陷阱 [8]'),
    ('2017', '陷阱态定位方法', '提出 LFN 噪声 + 瞬态测量 + TCAD 仿真联合定位陷阱态位置 [6]'),
    ('2021', '空穴再分布模型', '提出 hole redistribution 模型解释碳掺杂 buffer 中热激活 R_ON 应力/恢复瞬态（活化能 0.9 eV）[4]'),
    ('2021', '动态 R_ON 建模新方法', '提出高斯分布替代离散时间常数叠加的建模方法，降低模型复杂度 52% [3]'),
    ('2023', '陷阱态表征技术综述', '系统综述 GaN HEMT 陷阱态位置、能级、表征技术（体陷阱 vs 界面陷阱）[1]'),
]
for i, (year, milestone, contrib) in enumerate(data):
    table.cell(i+1, 0).text = year
    table.cell(i+1, 1).text = milestone
    table.cell(i+1, 2).text = contrib

# ===== 第3章 =====
doc.add_heading('第3章 核心进展与横向对比', level=1)

doc.add_heading('3.1 陷阱态类型与位置', level=2)
t2 = doc.add_table(rows=8, cols=5, style='Table Grid')
for j, h in enumerate(['陷阱态类型', '位置', '能级', '活化能', '主要影响']):
    t2.cell(0, j).text = h
rows2 = [
    ('表面陷阱态', 'AlGaN/GaN 界面、钝化层界面', '导带附近', '-', '电流坍塌、drain-lag'),
    ('界面陷阱态', 'AlGaN/GaN 异质结界面', 'Ev+0.84 eV', '0.84 eV', '动态 R_ON 增加'),
    ('体陷阱态（buffer）', 'GaN buffer', 'EC-0.57 eV', '0.57 eV', 'drain-lag、RF 功率退化'),
    ('体陷阱态（C 掺杂）', 'Carbon-doped buffer', 'Ev+0.9 eV', '0.9 eV', 'R_ON 应力/恢复瞬态'),
    ('体陷阱态（Fe 掺杂）', 'Fe-doped buffer', 'Ea=0.6 eV', '0.6 eV', '电流坍塌'),
    ('栅介质陷阱态', '栅介质层', '-', '-', 'V_th 亚稳态漂移'),
    ('位错相关陷阱态', 'Threading dislocations', '-', '-', '大时间常数瞬态（秒级）'),
]
for i, row in enumerate(rows2):
    for j, val in enumerate(row):
        t2.cell(i+1, j).text = val

doc.add_heading('3.2 陷阱态表征技术对比', level=2)
t3 = doc.add_table(rows=7, cols=5, style='Table Grid')
for j, h in enumerate(['表征技术', '适用陷阱态', '时间分辨率', '温度范围', '优点']):
    t3.cell(0, j).text = h
rows3 = [
    ('Pulsed I-V', '所有类型', 'μs-ms', '室温', '直接测量动态 R_ON、电流坍塌'),
    ('C-V 测量', '界面陷阱态', '-', '变温', '可提取界面态密度'),
    ('DLTS', '体陷阱态', 'ms-s', '变温', '精确测量能级和俘获截面'),
    ('LFN/RTN 噪声', '表面/界面陷阱态', 'μs-s', '室温', '可定位陷阱态位置'),
    ('瞬态测量', '所有类型', 'μs-s', '变温', '可测量陷阱态时间常数'),
    ('拉曼光谱', '应力/温度', '-', '变温', '非接触测量应力和温度'),
]
for i, row in enumerate(rows3):
    for j, val in enumerate(row):
        t3.cell(i+1, j).text = val

doc.add_heading('3.3 陷阱态建模方法对比', level=2)
t4 = doc.add_table(rows=6, cols=4, style='Table Grid')
for j, h in enumerate(['建模方法', '物理基础', '适用场景', '复杂度']):
    t4.cell(0, j).text = h
rows4 = [
    ('离散时间常数叠加', '经验模型', '小信号模型', '高'),
    ('高斯分布时间常数', '物理分布', '大信号模型', '低（降低 52%）'),
    ('SRH 统计模型', 'Shockley-Read-Hall', '大信号模型，偏置/温度依赖', '中'),
    ('空穴再分布模型', '空穴发射/再捕获', 'R_ON 应力/恢复瞬态', '中'),
    ('TCAD 仿真', '漂移扩散 + 自热', '多物理场耦合', '高'),
]
for i, row in enumerate(rows4):
    for j, val in enumerate(row):
        t4.cell(i+1, j).text = val

# ===== 第4章 =====
doc.add_heading('第4章 共识分歧与方法学评述', level=1)

doc.add_heading('4.1 学界共识', level=2)
consensus = [
    '陷阱态是导致 GaN HEMT 可靠性问题的核心机制 [1][2][4][5][7][8]',
    '陷阱态位置多样：表面、界面、体（buffer）、栅介质、位错 [1][5][6][7][8][9]',
    '陷阱态能级覆盖范围广：从 Ev+0.84 eV 到 EC-0.57 eV [4][5][7]',
    '陷阱态动态演化具有温度依赖性：热激活过程，活化能 0.57-0.9 eV [4][5][7][8]',
    'Pulsed I-V 是陷阱态表征的金标准 [1][5]',
]
for c in consensus:
    doc.add_paragraph(c, style='List Number')

doc.add_heading('4.2 学界分歧', level=2)
t5 = doc.add_table(rows=4, cols=3, style='Table Grid')
for j, h in enumerate(['分歧点', '观点 A', '观点 B']):
    t5.cell(0, j).text = h
rows5 = [
    ('陷阱态物理起源', '点缺陷（空位、间隙原子）', '位错相关缺陷'),
    ('R_ON 退化机制', '电子陷阱主导', '空穴再分布主导'),
    ('建模方法选择', '离散时间常数叠加', '高斯分布时间常数'),
]
for i, row in enumerate(rows5):
    for j, val in enumerate(row):
        t5.cell(i+1, j).text = val

doc.add_heading('4.3 方法学评述', level=2)
doc.add_paragraph(
    '实验方法：Pulsed I-V + 拉曼光谱 + 瞬态测量是主流组合，但缺少原位陷阱态密度监测手段。'
)
doc.add_paragraph(
    '仿真方法：TCAD 仿真已成熟，但陷阱态模型参数（密度、能级、俘获截面）需从实验拟合。'
)
doc.add_paragraph(
    '关联分析：多数研究仅关注单一陷阱态类型，缺少多陷阱态耦合演化的系统研究。'
)

# ===== 第5章 =====
doc.add_heading('第5章 空白与未来方向', level=1)

doc.add_heading('5.1 研究空白', level=2)
gaps = [
    '多物理场耦合下的陷阱态动态演化：现有研究多关注单一物理场（电或热），缺少电-热-应力耦合下的系统研究 [2][4][8]',
    '长期退化与陷阱态演化的关联：现有研究多为短期应力测试（分钟-小时级），缺少长期（月级）退化与陷阱态演化的关联分析 [4]',
    '陷阱态密度原位监测：缺少能在应力测试过程中实时监测陷阱态密度变化的手段',
    '多陷阱态耦合模型：现有模型多关注单一陷阱态类型，缺少表面+界面+体陷阱态耦合的综合模型',
]
for g in gaps:
    doc.add_paragraph(g, style='List Number')

doc.add_heading('5.2 未来研究方向', level=2)
directions = [
    '长期应力测试 + 阶段性陷阱态表征：在退化关键节点（2×、5×、10×）做 Pulsed I-V + 拉曼，建立宏观退化-微观陷阱态演化关联曲线',
    'TCAD 多物理场耦合仿真：搭建电-热-应力耦合 + 陷阱态模型的 TCAD 仿真，验证实验结果',
    '陷阱态密度拟合模型：从实验数据拟合陷阱态密度、能级、俘获截面参数，构建物理可信的陷阱态演化模型',
    '多陷阱态耦合机制研究：研究表面态、界面态、体陷阱态在电-热-应力耦合下的相互作用和竞争机制',
]
for d in directions:
    doc.add_paragraph(d, style='List Number')

# ===== 第6章 =====
doc.add_heading('第6章 参考文献', level=1)
refs = [
    '[1] ZOU Xiazhi, YANG Jiayi, QIAO Qifeng, et al. Trap Characterization Techniques for GaN-Based HEMTs: A Critical Review[J]. Micromachines, 2023, 14(11): 2044.',
    '[2] BELENIOTIS Petros, KRAUSE Sascha, ZERVOS Christos, et al. A Physics-Based Model for Slow Gate-Induced Electron Trapping in GaN HEMTs[J]. IEEE Transactions on Electron Devices, 2024, 71(7): 4058-4065.',
    '[3] WEISER Mathias C J, HUCKELHEIM Jan, KALLFASS Ingmar. A Novel Approach for the Modeling of the Dynamic ON-State Resistance of GaN-HEMTs[J]. IEEE Transactions on Electron Devices, 2021, 68(9): 4302-4309.',
    '[4] ZAGNI Nicolo, CHINI Alessandro, PUGLISI Francesco Maria, et al. "Hole Redistribution" Model Explaining the Thermally Activated RON Stress/Recovery Transients in Carbon-Doped AlGaN/GaN Power MIS-HEMTs[J]. IEEE Transactions on Electron Devices, 2021, 68(2): 697-703.',
    '[5] MENEGHINI M, BISI D, ROSSETTO I, et al. Trapping processes related to iron and carbon doping in AlGaN/GaN power HEMTs[C]// SPIE. 2015.',
    '[6] TARTARIN J G, SAUGNON D, LAZAR O, et al. Understanding traps locations and impact on AlGaN/GaN HEMT by LFN noise & transient measurements, and T-CAD simulations[C]// IEEE. 2017.',
    '[7] SASIKUMAR A, CARDWELL D W, AREHART A R, et al. Toward a physical understanding of the reliability-limiting EC-0.57 eV trap in GaN HEMTs[C]// IEEE IRPS. 2014.',
    '[8] BISI Davide, MENEGHINI Matteo, VAN HOVE Marleen, et al. Trapping mechanisms in GaN-based MIS-HEMTs grown on silicon substrate[J]. physica status solidi (a), 2015, 212(5): 1122-1129.',
    '[9] GHOSH Saptarsi, BAG Ankush, MUKHAPADHAY Partha, et al. Threading Dislocations in GaN HEMTs on Silicon: Origin of Large Time Constant Transients?[J]. 2015.',
    '[10] BENVEGNU Agostino. Trapping and Reliability Investigations in GaN-based HEMTs[J]. 2015.',
]
for ref in refs:
    p = doc.add_paragraph(ref)
    p.paragraph_format.space_after = Pt(4)

# ===== 保存 =====
import os
save_path = os.path.join(os.getcwd(), 'sa综述样例.docx')
doc.save(save_path)
print(f'文档已保存到: {save_path}')
print(f'文件大小: {os.path.getsize(save_path)} bytes')
