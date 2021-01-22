#!/usr/bin/env python
# coding: utf-8
"""
Copyright 2018 NAVER Corp.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import time
import math
import warnings
import tqdm
import torch
import tensorflow as tf
import numpy as np


warnings.filterwarnings("ignore")
config = tf.compat.v1.ConfigProto()
config.intra_op_parallelism_threads = 12
config.inter_op_parallelism_threads = 2
tf.compat.v1.Session(config=config)
# os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'


def asHHMMSS(s):
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m /60)
    m -= h *60
    return '%d:%d:%d'% (h, m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s<%s'%(asHHMMSS(s), asHHMMSS(rs))


#######################################################################
def sent2indexes(sentence, vocab):
    def convert_sent(sent, vocab):
        return np.array([vocab[word] for word in sent.split(' ')])
    if type(sentence) is list:
        indexes=[convert_sent(sent, vocab) for sent in sentence]
        sent_lens = [len(idxes) for idxes in indexes]
        max_len = max(sent_lens)
        inds = np.zeros((len(sentence), max_len), dtype=np.int)
        for i, idxes in enumerate(indexes):
            inds[i,:len(idxes)]=indexes[i]
        return inds
    else:
        return convert_sent(sentence, vocab)


def indexes2sent(indexes, vocab, eos_tok, ignore_tok=0): 
    '''indexes: numpy array'''
    def revert_sent(indexes, ivocab, eos_tok, ignore_tok=0):
        toks=[]
        length=0
        indexes=filter(lambda i: i!=ignore_tok, indexes)
        for idx in indexes:
            toks.append(ivocab[idx])
            length+=1
            if idx == eos_tok:
                break
        return ' '.join(toks), length
    
    ivocab = {v: k for k, v in vocab.items()}
    if indexes.ndim==1:# one sentence
        return revert_sent(indexes, ivocab, eos_tok, ignore_tok)
    else:# dim>1
        sentences=[] # a batch of sentences
        lens=[]
        for inds in indexes:
            sentence, length = revert_sent(inds, ivocab, eos_tok, ignore_tok)
            sentences.append(sentence)
            lens.append(length)
        return sentences, lens



from torch.nn import functional as F

use_cuda = torch.cuda.is_available()


def to_tensor(data):
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if use_cuda:
        tensor = tensor.cuda()
    return tensor


def gaussian_kld(recog_mu, recog_logvar):
    kld = -0.5 * torch.sum(1 + recog_logvar - recog_mu.pow(2) - recog_logvar.exp())
    return kld


SENTENCE_LIMIT_SIZE = 7
with open('data/vocab.txt') as vocab_file:
    vocab = vocab_file.read().strip().split('\n')
    # import pdb
    # pdb.set_trace()


# ### 构造映射

# In[30]:

# 单词到编码的映射，例如machine -> 10283
word_to_token = {word: token for token, word in enumerate(vocab)}
# 编码到单词的映射，例如10283 -> machine
token_to_word = {token: word for word, token in word_to_token.items()}

# ### 转换文本

# In[61]:

def convert_text_to_token(sentence, word_to_token_map=word_to_token, limit_size=SENTENCE_LIMIT_SIZE):
    """
    根据单词-编码映射表将单个句子转化为token

    @param sentence: 句子，str类型
    @param word_to_token_map: 单词到编码的映射
    @param limit_size: 句子最大长度。超过该长度的句子进行截断，不足的句子进行pad补全

    return: 句子转换为token后的列表
    """
    # 获取unknown单词和pad的token
    unk_id = word_to_token_map["<unk>"]
    pad_id = word_to_token_map["<pad>"]

    # 对句子进行token转换，对于未在词典中出现过的词用unk的token填充
    tokens = [word_to_token_map.get(word, unk_id) for word in sentence.lower()]

    # Pad
    if len(tokens) < limit_size:
        tokens.extend([0] * (limit_size - len(tokens)))
    # Trunc
    else:
        tokens = tokens[:limit_size]

    return tokens

word_to_vec = {}

# ### 构造词向量矩阵

# In[91]:

VOCAB_SIZE = len(vocab)  # 10384
EMBEDDING_SIZE = 300

# In[92]:

# 初始化词向量矩阵（这里命名为static是因为这个词向量矩阵用预训练好的填充，无需重新训练）
static_embeddings = np.zeros([VOCAB_SIZE, EMBEDDING_SIZE])

for word, token in tqdm.tqdm(word_to_token.items()):
    # 用glove词向量填充，如果没有对应的词向量，则用随机数填充
    word_vector = word_to_vec.get(word, 0.2 * np.random.random(EMBEDDING_SIZE) - 0.1)
    static_embeddings[token, :] = word_vector

# 重置PAD为0向量
pad_id = word_to_token["<pad>"]
static_embeddings[pad_id, :] = np.zeros(EMBEDDING_SIZE)

# In[93]:

static_embeddings = static_embeddings.astype(np.float32)

# 清空图
tf.compat.v1.reset_default_graph()

# In[31]:

# 定义神经网络超参数
HIDDEN_SIZE = 512
LEARNING_RATE = 0.001
EPOCHES = 50
BATCH_SIZE = 256

# In[32]:
# DNN:

model_name = 'dnn'

with tf.name_scope("dnn"):
    # 输入及输出tensor
    with tf.name_scope("placeholders"):
        inputs = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None, 7), name="inputs")
        targets = tf.compat.v1.placeholder(dtype=tf.int64, shape=(None), name="targets")

    # embeddings
    with tf.name_scope("embeddings"):
        # 用pre-trained词向量来作为embedding层
        embedding_matrix = tf.Variable(initial_value=static_embeddings, trainable=False, name="embedding_matrix")
        embed = tf.nn.embedding_lookup(embedding_matrix, inputs, name="embed")
        # 相加词向量得到句子向量
        sum_embed = tf.reduce_sum(embed, axis=1, name="sum_embed")

    # model
    with tf.name_scope("model"):
        # 隐层权重
        W1 = tf.Variable(tf.compat.v1.random_normal(shape=(EMBEDDING_SIZE, HIDDEN_SIZE), stddev=0.1), name="W1")
        b1 = tf.Variable(tf.zeros(shape=(HIDDEN_SIZE), name="b1"))

        # 输出层权重
        W2 = tf.Variable(tf.compat.v1.random_normal(shape=(HIDDEN_SIZE, 3), stddev=0.1), name="W2")
        b2 = tf.Variable(tf.zeros(shape=(1), name="b2"))

        # 结果
        z1 = tf.add(tf.matmul(sum_embed, W1), b1)
        a1 = tf.nn.relu(z1)

        logits = tf.add(tf.matmul(a1, W2), b2)
        predictions = tf.nn.softmax(logits, name="predictions")
        predict_value = tf.cast(tf.argmax(predictions, 1), tf.int64)

    # loss
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))

    # optimizer
    with tf.name_scope("optimizer"):
        optimizer = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # evaluation
    with tf.name_scope("evaluation"):
        correct_preds = tf.equal(tf.cast(tf.argmax(predictions, 1), tf.int64), targets)
        accuracy = tf.reduce_sum(tf.reduce_sum(tf.cast(correct_preds, tf.float32), axis=0))


def test_sentiment(predic_tokens):
    saver = tf.compat.v1.train.Saver()
    sentiments_count = {0: 0, 1: 0, 2: 0}

    with tf.compat.v1.Session() as sess:
        sentiment_result = []
        saver.restore(sess, "sent_predictor/dnn_checkpoints_small_dataset/dnn")
        # saver.restore(sess, "{}_checkpoints/dnn".format(model_name))
        for sentence in predic_tokens:
            input_tokens = np.array(sentence, dtype=np.int64).reshape(1, -1)
            pred_val = sess.run(predict_value, feed_dict={inputs: input_tokens})
            sentiment_result.append(pred_val[0])
            sentiments_count[pred_val[0]] += 1
        return sentiment_result, sentiments_count[0], sentiments_count[1], sentiments_count[2]


def write_predict_result_to_file(titles, predict_results, sentiment_result, output_file):
    assert len(predict_results) == len(sentiment_result) == 4 * len(titles)
    ptr = 0
    for i in range(len(titles)):
        output_file.write('{}\n'.format(titles[i]))
        for _ in range(4):
            output_file.write('{} {}\n'.format("".join([token_to_word[token] for token in predict_results[ptr]]),
                                               sentiment_result[ptr]))
            ptr += 1
        output_file.write('\n')


if __name__ == "__main__":
    words = '菊,秋菊萧萧黄叶黄,不知何处更堪愁,却向此中收拾得,却将归去被风瞒,春,春来不惜惜春华,春来无事可伤春,却被傍人夸得意,飞来飞去又飞飞,一支敕勒歌,一日长安万里馀,却向人间作队行,却向天涯问遗迹,空馀一曲水声中,送元二使安西,今日相逢不可亲,归来为我为重来,却向此中无觅处,夜来月落又还休,出塞,一夜无人月一毫,却将此地与谁论,却向天涯莫惆怅,只在沙头一点无,枫桥夜泊,不知何处不成阴,夜深月落月明中,月落月明如旧在,月明夜夜客愁多,游子吟,一生不见一毛锥,区区何用苦求人,不如二子如何有,空馀一粒吞他日,望洞庭,洞门深锁不成尘,洞中流水落花间,更有月明如此夜,不知此是非凡木,浪淘沙,一声渔笛引渔船,惊起沙鸥旧钓船,却来浪打鱼儿活,却是沙鸥旧所思,美,不是伤心第一筹,区区区区竟何如,不如二子如何苦,却将何事苦相关,梅,孤根孤顶两三枝,一点飞来点点尘,却被傍人描得得,却将风雨入寒梅,竹石,石壁遗踪迹已陈,谁与人间万古传,却向此中无用处,何如此去不相关,江上渔者,芦苇丛芦不见花,惊起梦魂惊梦梦,惊起芦花泊处船,芦荻满，芦荻花,石灰吟,一生灵隠几时回,空馀一点自生灵,莫怪空生无觅处,空馀空作千年计,乡,一年春事不堪悲,空馀一片在天涯,却向此中无觅处,空馀一片在沧洲,风,一生心事两相依,却被人言闲事了,若向此中无一事,今朝却被此生来,吹江南,一夜春风不肯休,月明如镜如眉颦,照人不见见龙颜,却是此中无可觅,春晓,晓来窗外雨声催,惊飞梦觉春残梦,惊起鸳鸯飞去梦,却来却被春风吹,池上,红榴花落绿成丛,一叶飞来一点春,桃李不堪风雨急,飞来不放绿杨枝,咏鹅,不是江村老病身,却嫌此地不能行,却向此中无此物,只缘无用可相忘,小儿垂钓,不知何事苦辛勤,若言不解为君子,何如何用苦求他,却向此中无一事,登鹳雀楼,楼上秋风吹笛声,吹笛愁人愁不到,却忆故乡春信尽,西风吹断楚江秋,咏柳,万缕千条拂画楼,半开半落半遮人,却忆西楼看不见,春风吹落几千春,一曲凉州词,长忆西风吹酒楼,今夜月明多少客,今夜满城无限思,却来骑马上阳台,江畔独步寻花,不惜春风不肯来,飞花飞去却飞来,不知春雨归何处,落花飞絮满江干,忆江南,忆君别我我为君,不堪归去更相思,却怜杜宇巴陵路,归来不为鲈鱼计,赠汪伦,一生一斛一尘埃,却向此中无此时,只恐此中无一事,只因却是一生心,望庐山瀑布,凿石为山凿石泉,此地无人可得名,却向此中无用宝,空馀一片落天间,早发白帝城,雨馀春水浸秋风,惊回一觉梦中愁,却忆江南归去路,回首空惊沙漠雁,春夜喜雨,雨馀蛙蚓鸣声滑,雨过山行半欲迷,却是江南归未得,却嫌归去却相容,唱绝句,一声杜宇啼鹃啼,春在春深处处啼,月落空庭月满庭,更有客愁无处着,黄鹤楼送孟浩然之广陵,孤舟夜泊古今州,回首平明隔断桥,遥想西风吹不起,却向此中还自去,渔歌子,渔郎不在钓鱼船,蓑笠渔蓑钓钓鱼,一蓑蓑笠蓑衣笠,轻舟轻桨去如飞,塞下曲,金马嘶风入楚宫,天子今朝又一年,却向前村收拾得,却愁人在五更啼,静夜思,夜深月落客星河,惊起梦魂飞去去,却是西风吹上客,夜来雨过寒窗下,古朗月行,月落霜风冷似冰,惊心不动玉花飞,月落月明天不去,月明照影还相忆,九月九日忆山东兄弟,一夜西风搅雨声,月明更有客愁多,今日不知何处去,却嫌西去又东来,望天门山,惆怅佳人旧日长,犹有旧时题柱杖,却是此生无用处,空馀一点自由来,出塞,老来无事可伤情,空馀一点哭声声,此去不知何处去,却来此地无他日,乐游原,春风吹断楚王家,一片飞来花下行,却忆西湖旧游子,今日又随流水去,清明,一生不作祢衡山,却将此地便安排,却向此中无用处,空馀一点与人同,山行,不知何处可伤情,却嫌花外更无情,春去春来无处觅,却随飞去去匆匆,寻隐者不遇,不见山翁斑笋斑,归来不见玉京尘,却忆当时刘弼手,可怜一部一枝春,所见,一声霹雳震轰雷,一日回头失却踪,却向此中还有意,却来此地不惺惺,墨,一生不作一杯酒,却愁风雨为春愁,却忆故乡千万里,却向此中无此君,村四月,夜深月落人家语,却愁人世有愁多,今日不知春去晚,却傍柳边多少年,春日,春风不肯放晴来,吹落春风吹不开,春在花光无处处,春光不到不知春,小池,江水平山一万竿,惊起沙鸥白鹭飞,却来却被渔樵乐,归来却被钓鱼翁,惠崇春江晚景,春来无处不春风,落花飞絮满江流,春去不知春已去,杜鹃啼断春归去,泊船瓜洲,江水无人不可攀,却向此中求得计,今日不知何处去,却是江湖无限思,芙蓉楼送辛渐,不惜金壶与世粧,区区空里觅花钿,今朝又被东风去,今日重来旧会稽,四时田园杂兴,山深浅浅水平铺,游子归来又一年,今日又添春又去,风雨满庭人不到,己亥杂诗,一年一度一阳生,今日始知非是别,空馀一点不揩磨,却向此中无一点,夏,一生一缕一丝纶,不知何用苦辛苦,却被东风催行止,吹落残花无数点,美,一夜春风一两枝,不知春在女郎来,却向此中收拾得,今朝又向此中来,菊,秋菊无心独自芳,更堪重到菊花开,却忆秋风明月夜,却疑此夕听寒雨,春,不惜春工巧不知,何须更作此花看,却是春风无处觅,却教桃李又春风,夏,病夫久矣不能行,老来无事可商量,却是天公为私意,区区何苦苦相攻,题林安邸,南北东西万里天,回望天涯地下行,空馀一片空飞去,不知此夕何人见,村居,山崦人家草树深,雨后桃花几点红,却恐被风吹不得,却教春雨到江干,元日,春风吹雨过平生,却傍山家劝酒杯,却嫌野老无人问,却将春色付谁人,示儿,不是花中一病翁,却将此意与谁论,只向此中无一点,却将底事当头空,饮湖上初晴后雨,雨过春来不可怜,雨馀春水涨平川,雨过田园无处避,雨馀蛙蚓鸣蛙声,题西林壁,一声啼鸟破春寒,惊断鸳鸯飞去人,却将白发生前事,却将归去却还家,江南春,春来无处着春工,轻轻轻薄薄烟轻,日日征人行路远,却来不及为山行,秋夜将晓出离门迎凉有感,秋风吹雨暗黄昏,更有残春在树头,却忆旧时添得雨,更堪枕上听蛩声,悯农,田田禾稼稻秧田,蚕妇耕桑稻田秧,田家蚕妇农耕妇,蚕妇妇馌任何如,夏日绝句,雨过春风不肯晴,雨声飞絮满庭前,天上有人来照影,水光山色翠光浮,别董大,莫言无语不言归,何如此处更无情,却向此中无觅处,今朝又了又还去,夜雨寄北,一生不见一潸然,今日还他天地流,却来此地无人会,空馀一片在天涯,送友人,不是伤心第一人,却将此地更相亲,若使老人无别事,莫言归去却相思,过华清宫,天上人间万事空,空馀一片落天间,却是旧时宫里许,不知何处更飞流,黄鹤楼,楼阁翬飞十二楼,天下无人见羽仪,却向天涯空下筯,不知何处更无尘,乌衣巷,雨馀蛙蚓鸣毛响,惊起沙鸥掠水飞,却来自有平生志,区区区区竟何为,里赠花卿,红颜白发混泥涂,今日重来见素馨,今日不知春已老,却嫌红旆少年场,回乡偶书,一声鶗鴂断肠声,飞来不觉春风里,惊飞蝴蝶梦惊回,东风不是东风恶,采莲曲,荷叶莲花菡萏中,蜻蜓来去水边行,惊飞不得鱼儿活,却将飞絮作鱼羹,白云泉,山泉石泉溉不流,水流洗尽人间世,却向此中无一物,空馀一片落沧洲,里暮江吟,江南江北路如霜,归棹渔舟泊夜船,惊心不觉春风里,惊起沙鸥一点愁,宿建德江,夜深灯火夜深深,惊起梦魂惊梦梦,惊起沙鸥眠不起,却忆旧时今夜泊,写七步诗,不是伤春病不知,空馀陋巷为谁愁,不是天公有时节,区区区区竟何为,相思,金缕毶毶泪垂垂,天下无人空自飞,却是此中无处觅,只因空有断人知,离思,一声啼鸟自飞飞,惊回春梦归何处,空馀一点红尘断,何人为我折杨花,诉衷情,一夜萧然一枕边,此生不是伤心事,今日却来还自愧,如今不作他乡念,山中送别,春风吹落桃花发,飞来不是去年来,今日相逢相问处,归来却是江南客,长歌行,惆怅春风不肯休,却嫌行路不曾多,却向此中求得老,空馀一点哭声声,劝学,一生心事不关心,空馀陋巷为谁论,可怜不及陈蕃事,何如何以却求归,唐诗宋词,万里烟霄路不迷,空有空山旧筑坛,不知月落今无数,此心还有此中来,风,春风吹落花飞尽,惊飞不到人间绿,风雨满庭人不到,却将飞絮点飞埃,雅,一丘一壑起悲风,却向天涯作画屏,却忆水边鱼跃上,却将飞絮作波涛,颂,一声霹雳一声嘶,大地生来不动尊,却向此中无一物,不知何处不相逢,人生,一生不作祢衡山,却将此地便安排,只恐空空无觅处,却将归去却相容,冬,一生病骨怯寒衣,夜来雨过暗空山,却被梅花吹落尽,却傍春风不肯归,旧,一年春事不堪悲,今日满空江水去,却被人人惆怅望,今日重来旧日游,红日,红粉匀花不可开,轻轻轻薄未全开,若也不知春色早,也教红粉扑花飞,奔马,一生不见鷅鹠沛,却将归去觐天机,却将水石为山去,却将归去却茫茫,伯乐,一夜春风一病翁,却愁人语不成眠,却忆当时有心在,今朝却被他人。,爱情,春风不肯放归家,空馀一片在春风,却被桃花流水去,春风不到旧时愁,华夏,金屋无人挂碧漪,半夜半天半点红,月落梨花飞不尽,春来不是一年春,柳絮,柳絮飞来絮欲飞,飞絮飞飞去不归,风絮飞飞飞絮落,飞絮飞飞落絮飞,'
    content = words.strip().split(',')
    predic_tokens = [convert_text_to_token(text) for text in content if len(text) == 7]
    _, neg, neu, pos = test_sentiment(predic_tokens)
    print(neg, neu, pos)