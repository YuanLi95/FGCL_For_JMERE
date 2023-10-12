
#encoding:utf-8
import openai
import json
import  jsonlines

# Prp_all = "改写一下，要求补充一些并符合经济学术语：\n"
# # Q_3 = "通过之前研究，本文发现经过网购市场的快速发展，很多城市老年人也逐渐加入或者完全将网购作为购物渠道。企业的相关营销的手段是与老年人网购行为之间存在密不可分的关系。企业想要利用或开拓老年人市场，就要做出改变，满足相关需求。因此本文结合之前的一些调研，希望给昆明本地的企业做出一些合理性的思路和建议，希望对未来企业的发展做出自己的一点贡献。"
# # Q_3 = "一直以来人们都认为老年人标签多数是“刻板的，保守的甚至是孤独的”。但随着社会的高速发展，经济水平的快速提高，一些城市老年人特别是有见识，有文化，有经济实力的老年人正在发生思想上和消费观的巨大变化。根据调研有一部分的老年群体不参与网购的主要原因是不了解，没有兴趣。因此为了改善老年人自身传统认识，改变其消息的态度是政府和相关企业必须考虑做的事。我们已经步入老年化时代，激发其对于网购的兴趣来提高老年人的生活质量，提升老年人自身的幸福感对维护社会稳定有重要意义。"
# # Q_3 = "电子设备的操作复杂对于已经参与网购的城市老年人来说有显著的负面影响，其中“流程繁琐”这个因素尤为突出。由于整个网购包括退货的过程较为复杂也影响了一部分老年人参与网购。对于调研的老年人来说，单是网上注册过程对于没有经验的老年人来需要耗费大量的时间。更多的网站在整个注册和下单以及退货的过程中都没有任何指导，这无形之中给老年人网购又增加了壁垒。这些都是现存网购媒介存在的问题。"
# # Q_3 = "相关企业推出的相关网购服务想受到更多的老年人的使用和欢迎，就必须要以下几个方面做出改变：（1） 操作简单化，尽可能设置老年人专用链接且要尽可能简单；（2）设计差异化，应该设计一些专门的UI或者APP去考虑老年人存在的生理障碍特别是视力问题。（3）内容专有化。在内容上应该针对老年人专门设计，贴近老年人的喜好，打造老年人喜欢购物体验。让老年人享受现代信息技术的成果，丰富他们的晚年生活。"
# Q_3 = "社会因素是影响城市老年网购行为的重要因素之一，尤其是子女们的影响。我国自古以来比较注重孝道，长辈们也比较信任子女们。因此企业和政府可以借助儿女等年轻群体作为老年消费者的桥梁，使得老年人可以更好的参与网购。具体而言，企业与相关机构可以通过倡导孝文化，来鼓励年轻子女来带动长辈们进行网购，并鼓励和帮助老年人们学习使用电子设备，网上购物，以及了解更多有趣的网上知识，丰富老年群体的晚年生活，进而发挥年轻群体在老年人网购行为中的积极作用。"
Prp_all = "Below is a paragraph from an academic paper. Polish the writing to meet the academic style, improve the spelling, grammar, clarity, concision and overall readability. When neccessary, rewrite the whole sentence. Furthermore, list all modification and explain the reasons to do so in markdown table.\n"
# Q_3 = "MAS proposes pointer-specific tagging to characterize internal associations between aspects and opinions by incorporating syntactic dependencies. Besides, a multi-task alignment scheme based on the pointer0seoecifice tagging is used to tackle capture redundant triples."
# Q_3 = "Besides, we also remove the full SA Transformer (SA-Trans) and degenerate it to be similar to GTS. "
# #
# Prp_all ="给定以下维度[年, 季, 月, 周, 日, 一级部门, 二级部门, 三级部门, 销售区域, 销售分部, 销售小组, 销售姓名, 销售经验, 项目子项, 一级科目,合同金额,销售金额,绩目标_季度,业绩目标完成率_季度], 给我生成10条类似的句子对："
# Q_3 = "原：今年广州的合同额\n 改：今年销售分部为广州的合同金额\n \n原：今年北方大区的销售额 \n改：今年销售区域为北方的销售金额 \n 原：今年Q1的业绩目标及其完成率 \n 改：今年一季度的业绩目标_季度和业绩目标完成率_季度"
# Prp_all = "给定以下三元组关系:\n奥巴马-妻子-米歇尔奥巴马，奥巴马-国家-美国，奥巴马-职位-总统，美国-属于-美洲"
# Q_3="将上述三元组转换为句子形式："

# Prp_all  ="Below is a paragraph from an academic paper. Polish the writing to meet the academic style, improve the spelling, grammar, clarity, concision and overall readability. When neccessary, rewrite the whole sentence."
# Q_3 = "A is an extended version of Li-unified \cite{Li2019}, which is a unified method to implement a two-layer stacked LSTM model to extract aspect items and their sentiment polarity. Peng et al. \cite{Peng2020} modified this method to Li-unified-R+ by additionally extracting opinion terms in the first stage and pairing the extracted terms in the second stage to generate triplets."
# # print(Prp_new)
# Prp_all = "你是经济学的博士，帮我按照以下类似风格:网络消费行为是当前热门和重点领域的研究方向。随着新技术不断涌现，网络消费行为的研究也逐步深入。尽管已经取得了一些成果，但目前的研究主要关注年轻人，需要进一步深入研究作为主要网络消费群体的年轻人，并且要注意到老年群体在现在和未来的增长迅速，因此需要加强对其网络消费行为的研究\n根据以下内容或者你自己补充从而总结影响消费的因素文献综述:"
# Q_3 = "消费者行为影响因素理论是指研究消费者行为的相关理论，它探究了导致消费者购买决策形成的因素，以及这些因素如何相互作用、影响消费者行为的过程。研究消费者行为影响因素的理论方面，不同的学者有不同的观点。其中，二因素论和三因素论是比较著名的理论（1）对于二因素论而言，其涵盖的观点一共有三种。在这中间，Jagdish N. Sheth ＆Barwari Mittal（2001）提出的二因素论认为，环境因素（例如经济、生态特征、技术、气候等）以及个人因素（比方说性别、文化以及群体、年龄、社会阶层、文化还有遗传特征跟种族）属于能够给消费者行为带来影响的主因。Nicosia（2002）析以及总结了购买行为模式变量，所获结论是：环境以及心理方面的因素，都会影响到消费者行为。最后Blacken（2003）指出使消费者行为受到影响的主因在于环境因素（比方说家庭、意见领袖、社会阶层以及文化还有另外一些参考群体）以及个人因素（比方说态度、个性、知识还有动机）。（2）三因素论是在二因素论的基础上发展而来。第一种观点，由Thunder（2001）提出他在二因素论的基础上将“营销”（涵盖的内容有产品、促销、价格以及分销渠道）当做对消费决策造成影响的关键性因素。随后L.G. Schiffman ＆L.L. Kanuk提出另一种观点。二者觉得动机、态度、人格以及知觉之类的因素会在很大程度上影响到消费者。立足于大环境方面而言，群体、文化、家庭以及阶层之类的方面同样会影响消费者。除此之外，他们还强调了营销刺激对消费者行为的重要影响。"
#
# Prp_all = "你是经济学博士，帮我简单扩展以下内容"
# Q_3 = "3.1.3 网络营销感知因素： 营销是从企业进行的一个因素衡量。没有争议的是：网上的促销会带来一定的销售量的增加，例如双十一购物节。基于相关调研，本文拟从销售渠道，产品本身（价格、质量）、促销手段等来进行考虑。"


Prp_all = "你是经济学博士，请帮我扩展以下内容（多80词），要求符合本科经济学毕业论文。"
Q_3 = "3.1.2 网络风险感知因素：互联网的普及为人们带来了巨大的便利，同时也带来了各种风险。其中最显著的表现是个人信息的不断泄露和私密信息的高度曝光。对于网购而言，经常出现商家延迟或不发货的现象，商品与宣传不符，存在严重的质量和安全问题。此外，相关研究人员指出，互联网购物存在常见的风险类型，包括时间、身体、经济、社会、功能和心理风险等。因此，本文将从网购流程的不同环节入手，对该变量进行风险性评估。"

Prp_all = "Below is a paragraph from an academic paper. Polish the writing to meet the academic style, improve the spelling, grammar, clarity, concision and overall readability. When neccessary, rewrite the whole sentence. Furthermore, list all modification and explain the reasons to do so in markdown table."
Q_3 = "where $W_o \in \mathbb{R}^{{d_h} \times ({d_y+d_h})}$ and $b_o \in \mathbb{R}^{{d_h}}$ are represented trainable weight matrix and bias for fusing the tag probability distribution with the hidden state respectively. The variable $\tilde o_{i,j}^{l - 1}$ is initialized using its word pair representation, which is obtained by concatenating the output representations of the \emph{i}-th and \emph{j}-th words."









openai.api_key = "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"
# openai.api_base = "https://chatgpt-chatgpt-qalzfpmbnt.us-west-1.fcapp.run/v1"
completion = openai.ChatCompletion.create(
model="gpt-3.5-turbo",
messages=[
    {"role":"system","content":Prp_all},
  {"role": "user", "content": Q_3}
        ]
)
# print(Prp_new)
messge_content = completion["choices"][0]["message"]["content"]
print(messge_content)
