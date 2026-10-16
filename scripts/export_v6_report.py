"""Export the adopted v6 method as a shareable PDF and standalone HTML.

Report-only entry point: no fitting, predictions or frozen model files change.
Requires reportlab; the training environment does not need this dependency.
"""
from pathlib import Path
import base64
import hashlib
import html
import json
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'experiments/v6'
PDF = ROOT / 'output/pdf/house_prices_v6_final_report.pdf'
HTML = ROOT / 'docs/house_prices_v6_final_report.html'
TEAL = colors.HexColor('#167667')
INK = colors.HexColor('#203346')
MUTED = colors.HexColor('#596b7c')
PALE = colors.HexColor('#edf6f3')
WIDTH = A4[0] - 84


def fonts():
    defaults = ['/System/Library/Fonts/STHeiti Light.ttc', '/System/Library/Fonts/STHeiti Medium.ttc']
    regular = os.environ.get('REPORT_CJK_FONT', defaults[0])
    bold = os.environ.get('REPORT_CJK_BOLD_FONT', defaults[1])
    pdfmetrics.registerFont(TTFont('CJK', regular, subfontIndex=0))
    pdfmetrics.registerFont(TTFont('CJKBold', bold, subfontIndex=0))
    pdfmetrics.registerFontFamily('CJK', normal='CJK', bold='CJKBold', italic='CJK', boldItalic='CJKBold')


def main():
    selection = json.loads((ART/'selection.json').read_text())
    online = json.loads((ART/'online_result.json').read_text())
    audit = json.loads((ART/'submission_audit.json').read_text())
    reproduction = json.loads((ART/'reproduction_audit.json').read_text())
    digest = hashlib.sha256((ROOT/'submission_v6.csv').read_bytes()).hexdigest()
    assert digest == audit['sha256'] == online['submission_sha256']
    assert online['public_log_rmse'] == .11212
    assert selection['weights_in_log_space'] == {'v5':1/3, 'physical_e16':1/3, 'quality_e4':1/3}
    fonts()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyCN', fontName='CJK', fontSize=9.7, leading=15.2, textColor=INK, spaceAfter=7, wordWrap='CJK'))
    styles.add(ParagraphStyle(name='HeadingCN', fontName='CJKBold', fontSize=12.5, leading=19, textColor=TEAL, spaceBefore=12, spaceAfter=7, keepWithNext=True, wordWrap='CJK'))
    styles.add(ParagraphStyle(name='TitleCN', fontName='CJKBold', fontSize=24, leading=32, textColor=INK, spaceAfter=8))
    styles.add(ParagraphStyle(name='SmallCN', fontName='CJK', fontSize=8.2, leading=12.5, textColor=MUTED, spaceAfter=5, wordWrap='CJK'))
    styles.add(ParagraphStyle(name='CellCN', fontName='CJK', fontSize=9, leading=13.7, textColor=INK, wordWrap='CJK'))
    styles.add(ParagraphStyle(name='CodeCN', fontName='CJK', fontSize=8.4, leading=13.5, textColor=INK, wordWrap='CJK', backColor=colors.HexColor('#f2f5f8'), borderPadding=9, spaceAfter=10))
    styles.add(ParagraphStyle(name='ScoreCN', fontName='CJKBold', fontSize=37, leading=46, textColor=TEAL))
    styles.add(ParagraphStyle(name='CenterCN', fontName='CJK', fontSize=9.2, leading=14, textColor=INK, alignment=TA_CENTER, wordWrap='CJK'))
    P = lambda s, name='BodyCN': Paragraph(s, styles[name])

    def table(rows, widths, header=True):
        data = [[P(str(c), 'CellCN') for c in row] for row in rows]
        obj = Table(data, colWidths=widths, hAlign='LEFT', repeatRows=1 if header else 0)
        commands = [('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
                    ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
                    ('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#dbe4e9'))]
        if header: commands += [('BACKGROUND',(0,0),(-1,0),PALE)]
        obj.setStyle(TableStyle(commands))
        return obj

    intro = '最终采用 v6：将已有传统模型集成与两种 TabPFN 特征表示组合，以不同预测方向的互补性降低误差。模型和提交文件保持 v6.0.0 原样；v6.1.0 补充实测成绩、报告和分享说明。'
    data_rows = [
        ['项目','本次使用的数据'],
        ['比赛','Kaggle House Prices: Advanced Regression Techniques'],
        ['官方数据','用户下载的官方 ZIP；train.csv、test.csv、data_description.txt、sample_submission.csv'],
        ['规模','训练 1,460 行；测试 1,459 行；79 个解释变量'],
        ['目标与标识','SalePrice 为美元房价；以 ln(SalePrice) 建模；Id 只用于对应提交行，不作特征'],
        ['缺失值','保留字符串 None 的类别含义；NA 读作缺失；模型内预处理只使用当前训练部分'],
    ]
    history = [('v1','.11969'),('v2','.11741'),('v3','.11770'),('v4','.11679'),('v5','.11595'),('v6','.11212')]
    story = [P('KAGGLE / HOUSE PRICES / FINAL v6','SmallCN'),P('Built with PriorLabs-TabPFN','SmallCN'),P('House Prices 最终方案','TitleCN'),
             P('线上成绩与数据','HeadingCN'),P('0.11212','ScoreCN'),
             P('Kaggle log-RMSE，越低越好。由用户于 2026-10-16 反馈；未通过 Kaggle 接口独立读取。','SmallCN'),
             P('较上一版 v5 的 0.11595 降低 <b>0.00383，约 3.30%</b>；较首版 0.11969 降低约 6.32%。用户已选择保留本版，停止继续优化。'),
             P(intro),P('数据来源与任务','HeadingCN'),table(data_rows,[90,WIDTH-90]),
             P('已经提交的版本成绩','HeadingCN'),table([['版本']+[x[0] for x in history],['线上 log-RMSE']+['0'+x[1] for x in history]],[91]+[(WIDTH-91)/6]*6),
             P('最终三路融合','HeadingCN')]
    model_cells = [[P('<b>1/3 · v5</b><br/>既有 ElasticNet、CatBoost、SVR 集成','CenterCN'),
                    P('<b>1/3 · physical</b><br/>住宅结构特征 TabPFN<br/>16 个内部成员','CenterCN'),
                    P('<b>1/3 · quality</b><br/>质量类别表示 TabPFN<br/>4 个内部成员','CenterCN')]]
    model_table = Table(model_cells,colWidths=[WIDTH/3]*3)
    model_table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),PALE),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),11),('BOTTOMPADDING',(0,0),(-1,-1),11),('LINEBEFORE',(1,0),(-1,-1),2,colors.white)]))
    story += [model_table,Spacer(1,9),P('log(price) = (z_v5 + z_physical + z_quality) / 3','CodeCN'),
              P('z 表示各路的对数房价预测，最后用 exp 转回房价。等价于三路价格的几何平均；16 与 4 是内部集成规模，不是两路在最终结果中的权重。','SmallCN'),PageBreak()]

    comparisons = selection['local_selected_blend_metrics']
    old = selection['v5_same_local_metrics']
    cv_rows = [['同折本地 log-RMSE','v5 参考','最终 v6'],
               ['全体 1,460 行',f"{old['all']:.6f}",f"{comparisons['all']:.6f}"],
               ['排除两套极端房屋，仅诊断',f"{old['ordinary']:.6f}",f"{comparisons['ordinary']:.6f}"],
               ['Normal 成交，1,198 行',f"{old['normal']:.6f}",f"{comparisons['normal']:.6f}"]]
    ablation = [['本轮候选，均与 v5 各半融合','全体','普通房屋','Normal'],
                ['原始特征 / 4 成员','0.115493','0.104159','0.084423'],
                ['住宅结构特征 / 4 成员','0.114577','0.103989','0.084223'],
                ['住宅结构特征 / 16 成员','0.114594','0.103996','0.084197'],
                ['训练侧删除两套极端房屋 / 4 成员','0.115488','0.104153','0.084423'],
                ['质量字段改为类别 / 4 成员','0.115046','0.103650','0.084322']]
    story += [P('方法与实验依据','TitleCN'),
              P('两种表示如何互补','HeadingCN'),
              P('<b>physical：</b>原始变量上增加总面积、完工面积、浴室数、房龄、车库龄、门廊面积、质量等级与面积交互等，共 112 列。将 MSSubClass 当类别，排除 Id 与 SalePrice。'),
              P('<b>quality：</b>保留原始 79 个变量，将 OverallQual、OverallCond 显式改成类别字符串，保留其他字段。它与结构特征路线形成不同的预测视角。'),
              P('<b>模型：</b>使用官方公开 TabPFN v2 回归权重，固定来源 revision 与 SHA-256。安装包版本为 tabpfn 9.2.0，模型权重版本为 v2，两者含义不同。v6 的训练上下文使用全部 1,460 行；没有额外微调预训练权重。'),
              P('<b>v5：</b>复用已有 ElasticNet、CatBoost 与 SVR 的集成方向。其 SVR 增量在大面积区域按训练面积分位数逐渐减弱，完整历史代码和预测一并保存。'),
              P('五组配置的受控比较','HeadingCN'),table(ablation,[WIDTH-228,76,76,76]),
              P('上表固定各半融合，便于比较特征和训练处理。删除极端训练样本几乎没有额外收益，未用于最终模型。最终再将两个互补 TabPFN 方向与 v5 各取 1/3。','SmallCN'),
              P('最终配方的同折对比','HeadingCN'),table(cv_rows,[WIDTH-172,86,86]),
              P('验证使用全体训练数据的 KFold 5 折（shuffle=True，seed=5509），与核验过的 v5 缓存严格匹配；未删除任何验证行。所有预处理在各折训练部分拟合。'),
              P('这些房屋和拆分已用于历史筛选，因此本地数值具有方案选择偏差，不是全新独立验证，也不能与线上分数直接混作同一个指标。','SmallCN'),PageBreak()]

    sources = [
        ('比赛与数据','https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques'),
        ('TabPFN 官方实现','https://github.com/PriorLabs/TabPFN'),
        ('官方 v2 回归权重与许可','https://huggingface.co/Prior-Labs/TabPFN-v2-reg'),
        ('项目最终文档版本','https://github.com/WANGDAKEKE/kaggle-houseprice/tree/v6.1.0'),
    ]
    story += [P('结果判断与复现','TitleCN'),P('为什么保留三路等权','HeadingCN'),
              P('除 0%/25%/50%/75%/100% 的粗权重外，增加自然的三路等权方案。三路等权在普通房屋与 Normal 成交上都优于 TabPFN 总权重 50% 和 75% 的方案。75% 的总体本地分数更低，但相对 50% 的新增收益约 96.9% 来自两套极端房屋，因此未选作最终提交。'),
              P('相对原 v5，分别去掉每组改善最大的五套房屋后，普通房屋和 Normal 组仍有正净收益。不过调权增量仍集中在少数房屋，不能据此保证每一组或新数据都会改善。线上实测 0.11212 是本次最终采用的依据，之后没有继续据此调参。'),
              P('复现与文件入口','HeadingCN'),
              table([['位置','用途'],['submission_v6.csv','最终已提交文件，1,459 行'],['src/ 与 scripts/','特征、模型、实验与预测入口'],['experiments/v6/fit_context.joblib','两路 TabPFN 的训练上下文与配置'],['experiments/v6/pretrained/','44 MB 公开权重、来源与许可'],['submission_v5.csv','v6 复用的历史融合预测'],['experiments/v6/online_result.json','用户反馈的线上成绩和文件对应关系'],['docs/SHARING_GUIDE.md','接收者上手、平台与分享说明']],[205,WIDTH-205]),
              P('已有 macOS ARM64 环境时，运行：','SmallCN'),
              P('sh scripts/run_tabpfn.sh scripts/predict_v6.py<br/>sh scripts/run_tabpfn.sh scripts/report_v6.py','CodeCN'),
              P('使用 Python 3.12。原环境锁为 requirements-tabpfn.lock.txt；非 Mac ARM64 的安装参考 requirements-tabpfn-portable.txt。原锁文件仍须保留，预测脚本会核对其哈希。非 MPS 环境当前会选 CPU，尚未做跨平台运行验证。'),
              P(f"已核验全部 1,459 行、Id 顺序与正数房价。两模型全量重载后，对相同预测方向换成最终权重进行核验，最大对数差为 {reproduction['max_abs_log_prediction_difference']:.3g}。GPU 浮点误差并非逐位完全一致。",'SmallCN'),
              P('提交文件 SHA-256：<br/>'+digest,'SmallCN'),P('来源与版本','HeadingCN')]
    for label,url in sources:
        story.append(P(f'{label}：<link href="{url}" color="#167667">{html.escape(url)}</link>','SmallCN'))
    story.append(P('Built with PriorLabs-TabPFN。公开权重的 LICENSE.txt 随代码包保留。模型与提交冻结在 v6.0.0；v6.1.0 仅补充线上结果、最终报告和分享材料。','SmallCN'))

    def footer(canvas, doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#dbe4e9'));canvas.line(42,33,A4[0]-42,33)
        canvas.setFont('CJK',8);canvas.setFillColor(MUTED)
        canvas.drawString(42,20,'House Prices v6 | 2026-10-16 | 最终线上 0.11212')
        canvas.drawRightString(A4[0]-42,20,str(doc.page));canvas.restoreState()
    PDF.parent.mkdir(parents=True,exist_ok=True);HTML.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(PDF),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=36,bottomMargin=46,
        title='House Prices v6 最终方案与结果 0.11212',author='WANGDAKEKE / House Prices')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)

    def html_table(rows):
        return '<table>'+''.join('<tr>'+''.join(f'<{tag}>{html.escape(str(x))}</{tag}>' for x in row)+'</tr>' for i,row in enumerate(rows) for tag in [('th' if i==0 else 'td')])+'</table>'
    figure=base64.b64encode((ART/'figures/comparison.png').read_bytes()).decode()
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>House Prices v6 最终报告 - 0.11212</title>
<style>body{{font:16px/1.85 -apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif;color:#203346;background:#f3f6f8;margin:0}}main{{max-width:1000px;margin:28px auto;padding:36px;background:white;border-radius:14px}}h1{{line-height:1.3}}h2{{color:#167667;margin-top:30px}}.score{{font-size:50px;color:#167667;font-weight:700}}.note{{color:#596b7c;font-size:14px}}.recipe,pre{{background:#edf6f3;padding:18px;border-radius:8px;overflow-wrap:anywhere;white-space:pre-wrap}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #dbe4e9;text-align:left}}th{{background:#edf6f3}}img{{max-width:100%}}a{{color:#167667}}code{{overflow-wrap:anywhere}}</style><main>
<p class="note">KAGGLE / HOUSE PRICES / 2026-10-16</p><h1>House Prices v6 最终方案</h1><div class="score">0.11212</div><p>最终采用的 Kaggle 线上 log-RMSE，越低越好。成绩由用户反馈，未通过 Kaggle 接口独立读取；较 v5 降低 3.30%，较首版降低 6.32%。</p><p>{intro}</p>
<h2>数据与目标</h2>{html_table(data_rows)}<p>主方案保留所有 1,460 条训练房屋和所有验证行。没有使用外部测试真实房价或公开答案文件。</p>
<h2>最终模型</h2><div class="recipe"><b>1/3 v5 + 1/3 住宅结构特征 TabPFN + 1/3 质量类别特征 TabPFN</b><br/>log(price) = (z_v5 + z_physical + z_quality) / 3<br/>最后 exp 转回房价，等价于三路价格的几何平均。</div>
<p><b>v5</b> 复用既有 ElasticNet、CatBoost 与 SVR 集成。<b>physical</b> 在原始字段上构造总面积、房龄、浴室、质量交互等共 112 列，使用 16 个内部成员。<b>quality</b> 保留原始 79 列，将 OverallQual/OverallCond 当类别处理，使用 4 个内部成员。两路在最终结果中各占 1/3，权重不按内部成员数量分配。</p>
<p>使用官方公开 TabPFN v2 回归权重，Python 包版本 tabpfn 9.2.0；固定 revision、checkpoint SHA-256、训练上下文与随机种子。Id 与 SalePrice 不作输入特征，MSSubClass 当类别，预处理在各折训练部分拟合。</p>
<h2>实验比较</h2>{html_table(ablation)}<p class="note">固定与 v5 各半融合，比较五组配置。训练侧删除极端房屋未带来额外收益，未用于主方案。</p><h2>最终配方的同折结果</h2>{html_table(cv_rows)}<img src="data:image/png;base64,{figure}" alt="匹配历史五折的本地误差比较">
<p>全部历史训练数据 KFold 5 折，shuffle=True、seed=5509；与保存的 v5 缓存按 Id 和折号匹配。该数据已用于多轮筛选，本地评价存在方案选择偏差，不能当新独立验证或与线上分数混用。</p>
<h2>选择依据与局限</h2><p>在粗权重比较之外，采用自然三路等权。其普通房屋和 Normal 误差优于 50%/75% TabPFN 权重。75% 虽总体更低，对半权方案的新增收益约 96.9% 来自两套极端房屋，因此未作最终提交。相对原 v5，各组去除最大五笔正收益后普通/Normal 仍有净收益；调权增量仍集中，不能保证每一组都改善。线上最终为 0.11212，之后停止优化。</p>
<h2>版本演进</h2>{html_table([['版本','线上 log-RMSE']]+[[x[0],'0'+x[1]] for x in history])}
<h2>复现</h2><p>代码包包含原始数据、模型上下文、公开预训练权重与许可、历史提交、完整代码、依赖锁与审计。Python 3.12；Mac ARM64 原环境锁为 requirements-tabpfn.lock.txt，其他系统参考 requirements-tabpfn-portable.txt 并保留原锁文件供核验。非 MPS 环境当前选 CPU，跨平台尚未实际验证。</p><pre>sh scripts/run_tabpfn.sh scripts/predict_v6.py\nsh scripts/run_tabpfn.sh scripts/report_v6.py</pre>
<p>1,459 行与官方 Id 顺序已核验。两模型全量重载并对相同方向换权后，最大对数差 {reproduction['max_abs_log_prediction_difference']:.3g}。实际过程保存于 experiments/v6/reproduction_audit.json。</p><p>最终提交 SHA-256：<code>{digest}</code></p>
<h2>来源与交付</h2><ul>{''.join(f'<li><a href="{url}">{label}</a></li>' for label,url in sources)}</ul><p>私有 GitHub 仓库需要所有者邀请接收者为协作者并接受邀请。直接转发代码 ZIP 或本报告无需 GitHub 登录。<b>Built with PriorLabs-TabPFN</b>；公开模型许可随代码保留。</p><p class="note">此 HTML 图片已内嵌，单独转发即可查看。更多逐折与逐房屋数据见代码包 experiments/v6/。</p></main></html>'''
    HTML.write_text(body,encoding='utf-8')
    print(json.dumps({'pdf':str(PDF),'standalone_html':str(HTML),'submission_sha256':digest},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
