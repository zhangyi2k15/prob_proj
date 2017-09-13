#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 15:01:03 2017

@author: zhangyi
"""
import pandas as pd
import numpy as np
from WindPy import *
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from scipy.signal import argrelextrema


w.start()
dict_list = {#'000300.SH':'沪深300：沪深300指数综合反映了A股市场整体走势，具有良好的市场代表性和流动性，行业分布以金融和能源板块为主，成分股业绩稳定、风险较低，估值具有优势，长期投资价值较高，具有典型的传统大盘蓝筹指数特征',
             '000906.SH':'中证800： 中证800指数综合反映市场行情',
             '000961.SH':'中证上游资源产业指数：按照中证指数公司对产业链的划分以及中证行业分类方法，将石油、天然气与消费用燃料、铝、黄金、多种金属与采矿等行业的股票归为上游资源产业股票。中证上游资源产业指数从中证800指数样本股中挑选规模大、具有上游产业特征的公司股票组成样本股',
             '399440.SZ':'国证钢铁：国证钢铁指数主要以2002年12月31日为基日，基点为1000 点。反映A股市场钢铁细分行业公司股票的整体表现。',
             'H30588.CSI':'中证证保：中证证券保险指数从保险与证券行业中依照日均总市值选取不超过50家上市公司作为样本股，反映证券与保险行业上市公司整体表现，为投资者提供投资标的。',
             '000018.SH':'180金融：上证180金融股指数从上证180指数中挑选银行、保险、证券和信托等行业的股票组成样本股，以反映上海证券市场的金融股走势。',
             '399808.SZ':'中证新能：中证新能源指数以2011年12月31日为基日，以该日收盘后所有样本股的调整市值为基期，以1000点为基点。选取样本空间中新能源相关行业的上市公司股票，可再生能源（核能、风能、太阳能、页岩气、生物质能、地热能和潮汐能）生产，以及能源应用、存储及交互设备（如锂电池、铅酸电池、充电桩、超级电容等），以及其他属于新能源相关行业的上市公司等。',
             '399914.SZ':'300金融：沪深300金融地产指数选取沪深300指数成分股中属于金融地产行业的全部股票构成该指数的成分股，以综合反映沪深300成分股中金融地产股票的整体表现情况。',
             #'399330.SZ':'深证100:深证100指数是中国证券市场第一只定位投资功能和代表多层次市场体系的指数，该指数由深圳证券交易所委托深圳证券信息公司编制维护，包含了深圳市场A股(含中小板、创业板)流通市值最大、成交最活跃的100只成份股。指数采用的编制方法兼顾了成分股的市值和流动性，提高了指数的市场代表性和流动性。',
             #'000016.SH':'上证50：上证50指数是根据科学客观的方法，挑选上海证券市场规模大、流动性好的最具代表性的50只股票组成样本股，以便综合反映上海证券市场最具市场影响力的一批龙头企业的整体状况。上证50指数是作为衍生金融工具（上证50ETF期权）基础的投资指数，其成分股亦均为沪港通投资标的。',
             '399412.SZ':'国证新能：国证新能指数以沪深A股市场中新能源和新能源汽车行业上市公司为备选范围，采用日均总市值比重、日均自由流通市值比重与日均成交金额比重1:1:1进行加权排序，结合上市公司基本面情况，选取排名靠前的70只股票构成样本股。',
             #'000021.SH':'180治理：上证180公司治理指数是从上证180指数与上证公司治理指数样本股并集中挑选100只规模大、流动性好的股票组成的样本股编制而成的指数。主要用来反映上证公司治理板块中前100只规模大、流动性、治理状况良好公司股票的走势，覆盖到证监会行业分类中所有的13个门类行业，从换手率和流动性指标来看，180治理指数流动性要优于上证50、中证等指数。',
             #'000903.SH':'中证100：中证100指数是从沪深300指数样本股中挑选规模最大的100只股票组成样本股，以综合反映沪深证券市场中最具市场影响力的一批大市值公司的整体状况。该指数成分股流动性好，可投资性强。',
             'H11136.CSI':'中国互联网 ：中证海外中国互联网指数选样范围包括以大陆以外市场为主要上市地的中国互联网公司，涵盖了涉及互联网软件、家庭娱乐软件、互联网零售、互联网服务以及移动互联网等互联网相关企业，选取符合流动性标准的证券作为样本，以市值加权进行计算，并设定15%的权重上限。',
             #'HSCAIT.HK':'恒生A股行业龙头：恒生A股行业龙头指数将A股市值排名前300位的大型及中型股经过流动性检测后，在恒生全部11种行业中，挑出每个行业里总市值、净利润和营业收入三个因素综合排名前5位的股票，成分股最多55只，最终编制成的恒生A股行业龙头指数风格稳健，抗波动性优势明显。',
             #'000905.SH':'中证500:综合反映了沪深证券市场内小市值公司的整体状况，其行业分布广泛均匀，其中不乏医药生物、电子、计算机、传媒、机械设备等新兴产业，具有巨大的长期投资价值。',
             #'000051.SH':'180等权：涵盖大中蓝筹的同时增强了对成长性的关注，减低大市值行业（如金融地产、能源等）的配置，同时相对超配小市值成长性行业（如消费、医疗、信息技术等）',
             '930599.CSI':'中证高装：中证高端装备制造指数选取通信设备，电气部件与设备，重型电气设备，工业机械，建筑、农用机械与重型卡车，航空航天与国防、电子设备制造商、石油与天然气设备与服务等行业的代表性股票作为样本股。',
             #'399429.SZ':'新丝路：反映丝绸之路经济带区域内相关上市公司的整体表现，刻画丝绸之路经济带区域发展特点。“新丝路”以沪深A股市场中注册地为陕西、甘肃、宁夏、青海与新疆5省的上市公司为备选范围。',
             #'000028.SH':'上证180成长：上证180成长指数依据3项财务指标：营业收入增长率、净利润增长率和内部增长率，从上证180指数的180只样本股中，每半年精选最具成长特征的60只股票。在计算营业收入和净利润增长率时，用过去3年的营业收入与净利润数据采用回归方法确定长期增长趋势。与上证180、上证50、沪深300等主要市场指数相比，上证180成长指数的成长因子在参比各指数中最优，体现出高营业收入增长率、高净利润增长率和高内部增长率的显著高成长特征；同时兼顾了低估值，即安全性高，实现了高成长与低估值的平衡。',
             '930653.CSI':'CS食品饮：中证食品饮料指数是中证公司发布的跟踪食品饮料行业股票的指数，该指数兼顾防御与成长',
             #'399348.SZ':'深证价值：深证300指数完全覆盖23个申万一级行业，行业分布更为均衡，深证300指数的权重可以较多分配给化工、医药和电子元器件等行业，具备较好的成长性和弹性，风格更偏中小盘，也是深证300指数涨幅领先沪深300指数的原因',
             #'000847.SH':'中证腾安：腾安指数即中证腾安价值100指数，由腾讯公司与北京济安金信合作开发，采用量化策略与专家评审相结合的方法，选择市场价格相对低估的100家上市公司股票为样本，并使用等权重的加权式。',
             #'000852.SH':'中证1000：中证1000指数成分股由中证800指数样本股之外规模偏小且流动性好的1000只股票组成，与沪深300和中证500等指数形成互补，成份股的平均市值及市值中位数较中小板指、创业板指、中证500指数相比都更小，小盘股特征更加鲜明，更能反映A股市场中小盘股的市场表现。',
             #'399337.SZ':'深证民营：深证民营价格指数挑选出的是深圳证券市场规模大、流动性好、具有代表性的100家民营上市公司股票，选样时综合考虑股票的流通市值及成交金额，能够较好的反映深圳证券市场上民营企业股票价格变动的趋势。民营企业属于非公有制经济范畴，其在我国经济中的重要性日益增加，有望在新一轮改革中享受到更多的红利，具有巨大的发展潜力。深证民营指数行业分布广泛，指数兼具成长性和防御性，具备长期投资价值。',
             'H30171.CSI':'运输指数：中证全指运输指数选取交通运输行业中依照日均成交金额、日均总市值由高到低排名，剔除成交金额排名后10% 、以及累积总市值占比达到 98% 以后的股票，并且保持剔除后股票数量不少于50只；行业内剩余股票构成相应行业指数的样本股。',
             #'399963.SZ':'中证下游：中证下游消费服务指数包含申万医药生物、申万金融服务、申万信息服务、申万食品饮料、申万商业贸易等行业个股，行业覆盖度广，整体流动性较好。相对沪深300，中证下游消费服务指数风险调整后收益更高，且在11年市场整体下跌中表现出较好抗跌性，进可攻退可守，随着未来经济结构转型消费升级以及人均收入稳步提升带来的巨大消费需求，消费板块中长期具备较大投资价值',
             '399967.SZ':'中证军工：中证军工指数反映沪深两市军工行业上市公司的整体表现，由十大军工集团控股的且主营业务与军工行业相关的上市公司以及其他主营业务为军工行业的上市公司作为指数样本。鉴于我国军工产业未来的发展空间巨大，中证军工指数具备长期投资价值。从指数的历史表现来看，中证军工指数具有典型的高风险高收益的特征，并且受事件驱动影响较大。',
             '000998.SH':'中证TMT：中证TMT产业主题指数，由TMT产业中规模和流动性较好的100只公司股票组成,反映A股上市公司中数字新媒体相关产业公司股票的走势,并为投资者提供新的投资标的。',
             #'000989.SH':'全指可选：中证全指可选消费指数，该指数旨在反映沪深两市A股中可选消费行业类股票的整体表现。全指是指数系列的基准指数，由若干指数成份股一起构成，以综合反映股市大中小盘公司的整体状况。',
             '000827.SH':'中证环保：中证环保产业指数是根据联合国环境与经济综合核算体系对于环保产业的界定方法，将符合资源管理、清洁技术和产品、污染管理的公司纳入环保产业主题，采用等权重加权方式，反映上海和深圳市场环保产业公司表现的指数。',
             #'HSCEI.HI':'国企指数：恒生中国企业指数由港交所上市H股中市值最大、流动性最好的40只个股组成，汇聚各行业龙头企业，涵盖金融、能源、地产、原材料、耐用消费品等行业，市场代表性好。该指数是典型的大盘蓝筹风格指数，拥有较高的股息率，长期收益稳定。此外由于制度差异，H股相较A股存在一定折价，整体估值低于A股，未来随着沪港通的实施，有利于提升两地资金的流动性，进一步提升该指数投资价值。',
             #'399958.SZ':'创业成长：中证创业成长指数样本由在创业板、中小板以及部分主板市场上市的具备创业和高成长特征的 100只股票组成，旨在刻画沪深两市创业和成长特征较为显著的中小型上市股票群体的整体表现。',
             #'399812.SZ':'养老产业：中证养老产业指数以中证全指为样本空间，选取消闲用品、酒店旅游、教育文化、药品零售、乳品、家用品、医药卫生、人寿保险等相关行业中市值最大的80只股票作为样本股，并以等权重加权。',
             #'399006.SZ':'创业板指：创业板指数成分股中医药生物、文化传媒、信息技术等新兴产业、高新技术企业占比高，未来随着经济结构进一步调整，相应扶持政策出台将对上述新兴产业利好，成长性突出。同时指数波动性和下行风险较大，显示高风险高收益特征。',
             #'':'iBoxx亚债中国指数由Markit指数公司（Markit Indices Limited）编制并发布，其成分券包括政府发行的以及由政府提供信用支持的机构发行的人民币债券，例如国债、央行票据、政策性金融债和地方政府债券等。',
             '000978.SH':'医药100：随着人口老龄化加速、用药需求不断释放、医疗卫生费用支出占GDP比例逐步提升以及未来医疗服务水平不断提高，医药行业未来增长空间较大，具备中长期投资价值。医药支出的刚性需求又使其在下跌行情中具备一定防御性。中证医药100指数是以沪深A股为样本空间，医药卫生行业和药品零售行业中挑选日均总市值前100位的公司股票组成样本股。成分股几乎涵盖了所有医药行业上市公司股票，市场代表性强，流动性较好。此外该指数设置等权重因子，采用派许加权编制而成，配置更均衡'
             #'NDX.GI':'NDX100：是美国纳斯达克100只最大型本地及国际非金融类上市公司组成的股市指数，以市值作基础，并以一些规则平衡较大市值股份造成的影响。',
             #'930654.CSI':'CS休闲娱：中证休闲娱乐指数选取媒体、家庭娱乐软件、酒店、餐馆与休闲、休闲设备与用品等行业的代表性公司作为样本股，以反映休闲娱乐类上市公司股票的整体表现。',
             #'SPX.GI':'标普500：标准普尔500指数是国际市场公认的美股风向标，它覆盖美国500家代表性上市公司，集中在市场的大盘股，占美国股票市场总市值的75%，包括苹果、谷歌、可口可乐、星巴克、宝洁、耐克等一大批中国投资人耳熟能详的跨国公司。经过08年金融危机以来的调整，美国经济逐步恢复（尽管有起伏），目前是较好地配置美股市场QDII基金的阶段，中长期投资价值明显。',
             #'S&P United States REIT ':'美国REIT：MSCI美国REITs指数是较具有市场代表性的美国REITs指数，该指数剔除了按揭类REITs股票（固定收益债券，受利率影响较大），主要反映权益类REITs的市场表现。REITs通过对房地产市场进行投资并经营管理以获取稳定收入，具有流动性高、抵御通货膨胀等特点，还能分散单一投资国内资本市场带来的系统风险，具备中长期投资价值。 ',
             #'399806.SZ':'环境治理：中证环境治理指数是由膜法水处理、固废处理、大气治理和节能再生等与环境治理相关的公司中选取不超过50只股票组成样本股，以反映上市公司中环境治理指数股票的走势，并为投资者提供新的投资标的。'}
             #'':'标普全球石油指数：标普全球石油指数（S&P Global Oil Index）由全球最大的 120 家满足特定可投资性条件的石油和天然气行业上市公司组成， 旨在跟踪全球涉及油气勘探、 冶炼和生产业务的公司。'
             }
list_all = list(dict_list.keys())
close_all = pd.DataFrame()
for name in list_all:
    data = w.wsd(name, "open,high,low,close,volume,amt", "2005-01-01", "2017-09-12", "Fill=Previous")
    df = pd.DataFrame(data.Data, index=data.Fields, columns=pd.to_datetime(data.Times).date).T
    close_all[name] = df['CLOSE']
    df.to_csv(r'C:\Users\LatteMachine\Desktop\shangzheng\\'+name+'.csv', encoding='utf-8')
close_all.to_csv(r'C:\Users\LatteMachine\Desktop\shangzheng\close_all.csv')

######################################################
#计算收益率序列的相关系数
close_all = pd.read_csv(r'C:\Users\LatteMachine\Desktop\shangzheng\close_all.csv')
close_all.rename(columns={close_all.columns[0]:'date'},inplace=True)
close_all['date'] = pd.to_datetime(close_all['date'])
close_all.set_index ('date',inplace=True)

close_all_rt = close_all.pct_change(periods=1)[1:]
#close_rt_1yr = close_all_rt.loc[datetime(2012,1,1).date():]
close_rt_1yr = close_all_rt
mu_close = close_rt_1yr.mean()
sigma_close = close_rt_1yr.std()
#close_all_coef = close_all.corr(method='pearson')
#close_all_rt_coef = close_all_rt.corr()

#######
#根据相关系数，做假设检验，验证不同收益率序列之间是否具有强相关性，这一步我们，请参考cal_coef 函数


##归类，我们将强相关的函数归为一类，然后对每一类进行分析，
class_dict = {'class_1':['000961.SH', '399440.SZ', 'H30588.CSI', '000018.SH',
       '399808.SZ', '399914.SZ', '399330.SZ', '000016.SH', '399412.SZ',
       '000021.SH', '000903.SH', '000905.SH', '000051.SH', '930599.CSI',
       '399429.SZ', '000028.SH', '930653.CSI', '399348.SZ', '000847.SH',
       '000852.SH', '399337.SZ', 'H30171.CSI', '399963.SZ', '000989.SH',
       '000827.SH', '399958.SZ', '399812.SZ','399967.SZ','000998.SH',
       '399006.SZ','000978.SH','399806.SZ'],
    'class_2':['H11136.CSI'],'class_3':['HSCEI.HI'],'class_4':'NDX.GI',}



'''
选择最近一个月的数据作为


begin = datetime(2017,7,1).date()
end = datetime(2017,7,31).date()
close_7 = close_all.loc[begin:end]
del begin, end
close_7_rt = close_7.pct_change(periods=1)
close_7_rt = close_7_rt[1:]
coef = close_7_rt.corr()

###分类，将相关性高的数据归为一类
fig,ax = plt.subplots()
cax = ax.matshow(coef,vmin=-1,vmax=1)
fig.colorbar(cax)
#ticks = range(9)
#ax.set_xticks(ticks)
#ax.set_yticks(ticks)
names = list(coef.columns)
#ax.set_xticklabels(['']+names)
#ax.set_yticklabels(['']+names)

##t-test 验证相关性
def cal_coef(rt):
    #检验收益率序列之间是否具有相关性，采用student-t检验，并通过p-value来判断，p-value 越小，则拒绝原假设，繁殖则接受原假设。
    #注意，student-t检验的是两者的线性相关性，不能检验非线性相关性，飞线性相关性可通过spearman秩相关性检验，这里我们先考虑线性关系。
    length = rt.shape[1]
    tp = pd.DataFrame(index=rt.columns,columns=rt.columns)
    for i in range(length):
        for j in range(length):
            if i!=j:
                t,p = ttest_rel(rt.iloc[:,i],rt.iloc[:,j])
                #t = round(t,2)
                p = round(p,2)
                tp.iloc[i,j] = p
            else:
                tp.iloc[i,j] = 0
    #tp.fillna(0,inplace=True,axis=None)
    return tp
tp = cal_coef(close_7_rt)
'''
#归类，对强相关的数据归为一类，强弱是一个相对的标准，这里我们定义一个阈值threshold，如果abs（相关系数）>threshold，
#则我们称两列序列强相关，否则为弱相关。
def cluster(coef, threshold=0.9):
    cluster_strings = {}
    coef_list = list(coef.columns)
    coef_res = coef
    k = 1
    while coef_list:
        cluster_strings['type_'+str(k)] = list(coef_res.index[coef.iloc[:,1]>threshold])
        coef_list = list(coef_res.index[coef_res.iloc[:,k]<=threshold])
        coef_res = coef_res.loc[[coef_list,coef_list]]
        k+=1
    return cluster_strings





cluster_strings = cluster(coef,0.8)

###########################################
##排序,对所有的相关性进行排序
sort_value = dict()
for i in range(len(coef)):
    for j in range(i+1,len(coef)):
        sort_value[coef.columns[i]+'_'+coef.columns[j]] = coef.iloc[i,j]
sort_value = pd.DataFrame(sort_value, index=['coef']).T
sort_value.sort_values(by=['coef'], ascending=False,inplace=True)
sort_value.plot(style=':.',color='b',title='rank of coef')

#mu_close 和sigma_close 分别表示收益率序列的均值和标砖差

close_rt_1yr_coef = close_rt_1yr.corr()
fig,ax = plt.subplots()
cax = ax.matshow(close_rt_1yr_coef,vmin=-1,vmax=1)
fig.colorbar(cax)
plt.title('coef matrix of return rate (last one year)')
plt.savefig('coef_matshow.png',dpi=400,bbox_inches='tight')



fig,axis = plt.subplots()
plt.plot(mu_close, sigma_close,'o',color='r')
plt.title('$\mu$ and $\sigma$')
plt.xlabel('$\mu$')
plt.ylabel('$\sigma$')
plt.savefig('mu_sigma.png',pdi=400,bbox_inches='tight')

#close_div = close_all.loc[datetime(2012,1,1).date():].div(close_all['000906.SH'][datetime(2012,1,1).date():],axis=0)
close_div = close_all.div(close_all['000906.SH'],axis=0)

def plot_fig(close_div, page_num=4, order=60):
    #plot close_div and scatter max point
    data = np.array(close_div)
    time_dict = {}
    for k in range(close_div.shape[1]//page_num):
        fig= plt.figure()
        for i in range(page_num*k,page_num*(k+1)):
            ax = fig.add_subplot(np.sqrt(page_num),np.sqrt(page_num),i-page_num*k+1)
            plt.plot(close_div[close_div.columns[i]])
            for label in ax.get_xticklabels():
                label.set_rotation(30)
                label.set_horizontalalignment('center')
            '''
            if i<np.sqrt(page_num)+page_num*k:
                plt.xticks([])
            else:
                pass
            '''
            plt.title(close_div.columns[i]+'/000906.SH')
            data_test = data[:,i]
            pos = argrelextrema(data_test, comparator=np.greater,order=order)
            #close_div[close_div.columns[i]].plot(ax=ax)
            plt.scatter(close_div.index[pos],close_div[close_div.columns[i]].iloc[pos],color='r')
            time_dict[close_div.columns[i]] = pd.DataFrame(close_div.index[pos])
            ma_5 = close_div[close_div.columns[i]].rolling(window=5).mean()
            ma_10 = close_div[close_div.columns[i]].rolling(window=10).mean()
            ma_5.plot(color='b')
            ma_10.plot(color='r')
       #plt.savefig(close_div.columns[i]+'_derived.png',dpi=400,bbox_inches='tight')
       # plt.xlabel('time')
       # plt.ylabel('ratio')
    if close_div.shape[1]%page_num:
        fig = plt.figure()
        for i in range(close_div.shape[1]%page_num):
            ax = fig.add_subplot(np.sqrt(page_num),np.sqrt(page_num),i+1)
            plt.plot(close_div[close_div.columns[i+close_div.shape[1]//page_num*page_num]])
            for label in ax.get_xticklabels():
                label.set_rotation(30)
                label.set_horizontalalignment('center')
            plt.title(close_div.columns[i+close_div.shape[1]//page_num*page_num]+'/HS300(last one year)')
            data_test = data[:,i+close_div.shape[1]//page_num*page_num]
            pos = argrelextrema(data_test, comparator=np.greater,order=order)
            close_div[close_div.columns[i+close_div.shape[1]//page_num*page_num]].plot(ax=ax)
            plt.scatter(close_div.index[pos],close_div[close_div.columns[i+close_div.shape[1]//page_num*page_num]].iloc[pos],color='r')
            time_dict[close_div.columns[i+close_div.shape[1]//page_num*page_num]] = pd.DataFrame(close_div.index[pos])
    else:
        pass

    #plt.savefig(close_div.columns[i]+'_derived.png',pdi=400,bbox_inches='tight')
   # plt.xlabel('time')
   # plt.ylabel('ratio')
    return time_dict
time_dict = plot_fig(close_div)
###分析divided之后的数据的收益率序列和原指数的收益率序列的关系，首先查看收益率的直方图之间是否有差距
#close_div_rt 表示两种指数相除之后的收益率序列，mu_div 表示该序列的均值sigma_div表示其标准差。
#sigma_idx是同一指数的相关性的

close_div_rt = close_div.pct_change()[1:]
mu_div = close_div_rt.mean()
sigma_div = close_div_rt.std()
sigma_idx = sigma_div.div(sigma_close)
sigma_idx.sort_values(inplace=True)
close_div_rt = close_div_rt[sigma_idx.index]
close_rt_1yr = close_rt_1yr[sigma_idx.index]

def plot_hist(close_div_rt,close_rt_1yr,sigma_idx, page_num=4):
    for k in range(close_div_rt.shape[1]//page_num):
        fig = plt.figure()
        for i in range(page_num*k,page_num*(k+1)):
            ax = fig.add_subplot(np.sqrt(page_num),np.sqrt(page_num),i-page_num*k+1)
            #ax.set_ylim(0,50)
            close_rt_1yr[close_rt_1yr.columns[i]].hist(bins=50,grid=False,ax=ax,color='r')
            #ax.text(close_div_rt[close_div_rt.columns[i]].quantile(0.02),200,'$\mu_{derived}$=%.4f, $\sigma_{derived}=%.4f$' %(mu_div[i],
                     #sigma_div[i]))
            close_div_rt[close_div_rt.columns[i]].hist(bins=50,grid=False,alpha=0.7,ax=ax,color='g')
            #ax.text(close_rt_1yr[close_rt_1yr.columns[i]].quantile(0.02),40,'$\mu_{original}$=%f, $\sigma_{original}=%f$' %(mu_close[i],
                     #sigma_close[i]))
            ax.text(close_rt_1yr[close_rt_1yr.columns[i]].quantile(0.02),150,'$\\frac{\sigma_{derived}}{\sigma_{original}}$=%.4f' %(sigma_idx[i]))
        #plt.savefig('close_div_rt_'+str(k)+'.png',pdi=400, bbox_inches='tight')
    if close_div_rt.shape[1]%page_num:
        fig = plt.figure()
        for i in range(close_div_rt.shape[1]%page_num):
            ax = fig.add_subplot(np.sqrt(page_num),np.sqrt(page_num),i+1)
            #ax.set_ylim(0,50)
            close_rt_1yr[close_rt_1yr.columns[i]].hist(bins=50,grid=False,ax=ax,color='r')
            #ax.text(close_div_rt[close_div_rt.columns[i]].quantile(0.02),45,'$\mu_{derived}$=%f, $\sigma_{derived}=%f$' %(mu_div[i], sigma_div[i]))
            close_div_rt[close_div_rt.columns[i+36]].hist(bins=50, grid=False,aipha=0.7,ax=ax,color='g')
            #ax.text(close_rt_1yr[close_rt_1yr.columns[i]].quantile(0.02),40,'$\mu_{original}$=%f, $\sigma_{original}=%f$' %(mu_close[i], sigma_close[i]))
            ax.text(close_rt_1yr[close_rt_1yr.columns[i]].quantile(0.02),150,'$\\frac{\sigma_{derived}}{\sigma_{original}}$=%.4f' %(sigma_idx[i+36]))
        #plt.savefig('close_div_rt_'+str(i+4)+'.png',pdi=400, bbox_inches='tight')


### 找到最大值和最小值
'''
def find_max(close_div, periods=125):
    idx_max = pd.DataFrame(columns=close_div.columns)
    for i in range(periods, close_div.shape[0]-periods,1):
        idx_max = idx_max.append(close_div.iloc[i-periods:i+periods].idxmax(),ignore_index=True)
    return idx_max

idx_max = find_max(close_div)
'''

#def find_max(close_div,order=60):
    #=注意到data_test是一个dataframe，首先需要将其转化为ndarray








































