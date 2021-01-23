import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from helper import to_tensor, gaussian_kld
from modules import Encoder, VAE, Decoder


class multiVAE(nn.Module):
    def __init__(self, config, vocab, rev_vocab, PAD_token=0):
        super(multiVAE, self).__init__()
        assert rev_vocab['<pad>'] == PAD_token
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.embed_size = config.emb_size
        self.hidden_size = config.n_hidden
        self.bow_size = config.bow_size
        self.rev_vocab = rev_vocab
        self.dropout = config.dropout
        self.go_id = self.rev_vocab["<s>"]
        self.eos_id = self.rev_vocab["</s>"]
        self.maxlen = config.maxlen
        self.clip = config.clip
        self.temp = config.temp
        self.full_kl_step = config.full_kl_step
        self.z_size = config.z_size
        self.init_w = config.init_weight
        self.softmax = nn.Softmax(dim=1)
        self.bidirectional = config.bidirectional
        self.lr_ae = config.lr_ae
        self.lr_vae = config.lr_vae

        # 如果LSTM双向，则两个方向拼接在一起
        self.encoder_output_size = self.hidden_size * (1 + int(self.bidirectional))
        # 标题和首句拼接在一起，
        self.context_dim = self.encoder_output_size * 2
        self.decoder_input_size = self.z_size

        # build components
        self.layers = nn.ModuleDict()
        self.layers["embedder"] = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=PAD_token)
        # 对title, 每一句诗做编码, 默认双向LSTM，将最终的一维拼在一起
        self.layers["seq_encoder"] = Encoder(embedder=self.layers["embedder"], input_size=config.emb_size, hidden_size=config.n_hidden,
                                   bidirectional=self.bidirectional, n_layers=config.n_layers, noise_radius=config.noise_radius)

        # 先验网络
        self.layers["neg_vae"] = VAE(target_size=self.encoder_output_size, z_size=self.z_size, dropout=self.dropout, init_weight=self.init_w)
        self.layers["neu_vae"] = VAE(target_size=self.encoder_output_size, z_size=self.z_size, dropout=self.dropout, init_weight=self.init_w)
        self.layers["pos_vae"] = VAE(target_size=self.encoder_output_size, z_size=self.z_size, dropout=self.dropout, init_weight=self.init_w)

        # 词 Bow loss
        self.layers["bow_project_pos"] = nn.Sequential(
            nn.Linear(self.decoder_input_size, self.bow_size),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.bow_size, self.vocab_size)
        )
        self.layers["bow_project_neu"] = nn.Sequential(
            nn.Linear(self.decoder_input_size, self.bow_size),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.bow_size, self.vocab_size)
        )
        self.layers["bow_project_neg"] = nn.Sequential(
            nn.Linear(self.decoder_input_size, self.bow_size),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.bow_size, self.vocab_size)
        )

        # self.layers["decoder"] = Decoder(embedder=self.layers["embedder"], input_size=self.embed_size,
        #                        hidden_size=self.hidden_size,
        #                        vocab_size=self.vocab_size, n_layers=1)

        self.layers["vae_decoder_pos"] = Decoder(embedder=self.layers["embedder"], input_size=self.embed_size,
                           hidden_size=self.hidden_size,
                           vocab_size=self.vocab_size, n_layers=1)
        self.layers["vae_decoder_neu"] =  Decoder(embedder=self.layers["embedder"], input_size=self.embed_size,
                           hidden_size=self.hidden_size,
                           vocab_size=self.vocab_size, n_layers=1)
        self.layers["vae_decoder_neg"] = Decoder(embedder=self.layers["embedder"], input_size=self.embed_size,
                           hidden_size=self.hidden_size,
                           vocab_size=self.vocab_size, n_layers=1)

        self.layers["init_decoder"] = nn.Sequential(
            nn.Linear(self.decoder_input_size + self.context_dim, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size, eps=1e-05, momentum=0.1),
            nn.LeakyReLU()
        )

        self.layers["init_decoder_hidden"] = nn.Sequential(
            nn.Linear(self.decoder_input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size, eps=1e-05, momentum=0.1),
            nn.LeakyReLU()
        )

        self.layers["init_decoder_hidden"].apply(self.init_weights)
        self.layers["bow_project_neg"].apply(self.init_weights)
        self.layers["bow_project_neu"].apply(self.init_weights)
        self.layers["bow_project_pos"].apply(self.init_weights)

        # self.optimizer_AE = optim.AdamW(list(self.layers["embedder"].parameters())
        #                                 + list(self.layers["seq_encoder"].parameters())
        #                                 + list(self.layers["vae_decoder_pos"].parameters())
        #                                 + list(self.layers["vae_decoder_neu"].parameters())
        #                                 + list(self.layers["vae_decoder_neg"].parameters())
        #                                 + list(self.layers["init_decoder"].parameters()), lr=self.lr_ae)
        self.optimizer_AE = {
            'pos': optim.AdamW(list(self.layers["embedder"].parameters())
                               + list(self.layers["init_decoder"].parameters())
                               + list(self.layers["seq_encoder"].parameters())
                               + list(self.layers["vae_decoder_pos"].parameters())
                               + list(self.layers["pos_vae"].parameters())
                               + list(self.layers["bow_project_pos"].parameters()), lr=self.lr_vae),
            'neu': optim.AdamW(list(self.layers["embedder"].parameters())
                               + list(self.layers["init_decoder"].parameters())
                               + list(self.layers["seq_encoder"].parameters())
                               + list(self.layers["vae_decoder_neu"].parameters())
                               + list(self.layers["neu_vae"].parameters())
                               + list(self.layers["bow_project_neu"].parameters()), lr=self.lr_vae),
            'neg': optim.AdamW(list(self.layers["embedder"].parameters())
                               + list(self.layers["init_decoder"].parameters())
                               + list(self.layers["seq_encoder"].parameters())
                               + list(self.layers["vae_decoder_neg"].parameters())
                               + list(self.layers["neg_vae"].parameters())
                               + list(self.layers["bow_project_neg"].parameters()), lr=self.lr_vae)
        }

        self.optimizer_VAE = {
            'pos': optim.AdamW(list(self.layers["vae_decoder_pos"].parameters())
                               + list(self.layers["pos_vae"].parameters())
                               + list(self.layers["bow_project_pos"].parameters()), lr=self.lr_vae),
            'neu': optim.AdamW(list(self.layers["vae_decoder_neu"].parameters())
                               + list(self.layers["neu_vae"].parameters())
                               + list(self.layers["bow_project_neu"].parameters()), lr=self.lr_vae),
            'neg': optim.AdamW(list(self.layers["vae_decoder_neg"].parameters())
                               + list(self.layers["neg_vae"].parameters())
                               + list(self.layers["bow_project_neg"].parameters()), lr=self.lr_vae)
        }

        # self.lr_scheduler_AE = optim.lr_scheduler.StepLR(self.optimizer_AE, step_size=10, gamma=0.6)

        self.criterion_ce = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        self.reconstruct_loss = dict()

    def set_full_kl_step(self, kl_full_step):
        self.full_kl_step = kl_full_step

    def force_change_lr(self, new_init_lr_ae):
        self.optimizer_AE = optim.Adam(self.parameters(), lr=new_init_lr_ae)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.uniform_(-self.init_w, self.init_w)
            m.bias.data.fill_(0)

    # 按照输入的sentiment_label，从对应的高斯分布上面采样
    def sample_from_specific_latent_area(self, sentiment_label):
        batch_size = sentiment_label.size(0)
        z_origin = to_tensor(torch.randn([batch_size, self.z_size]))

        # 使用sentiment_label mask掉每一个未使用当前情感的句子，最后再组合即得到所有采样
        mask_pos = sentiment_label.gt(1).view(-1, 1).expand(batch_size, self.z_size)
        mask_neu = sentiment_label.eq(1).view(-1, 1).expand(batch_size, self.z_size)
        mask_neg = sentiment_label.lt(1).view(-1, 1).expand(batch_size, self.z_size)

        z_pos = z_origin.mul(mask_pos)
        z_neu = z_origin.mul(mask_neu)
        z_neg = z_origin.mul(mask_neg)
        return {'pos': z_pos, 'neu': z_neu, 'neg': z_neg}

    def train_VAE(self, global_t, target, target_lens, sent_name):
        self.train()
        self.optimizer_VAE[sent_name].zero_grad()

        target_hidden, _ = self.layers["seq_encoder"](target[:, 1:], target_lens - 1)
        mu, logsigma, z_post = self.layers["{}_vae".format(sent_name)](target_hidden)

        output = self.layers["vae_decoder_{}".format(sent_name)](init_hidden=self.layers["init_decoder_hidden"](z_post), context=None, inputs=target[:, :-1])
        flattened_output = output.view(-1, self.vocab_size)
        dec_target = target[:, 1:].contiguous().view(-1)

        mask = dec_target.gt(0)  # 即判断target的token中是否有0（pad项）
        masked_target = dec_target.masked_select(mask)  # 选出非pad项
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)  # [(batch_sz * seq_len) x n_tokens]
        masked_output = flattened_output.masked_select(output_mask).view(-1, self.vocab_size)
        self.rc_loss = self.criterion_ce(masked_output / self.temp, masked_target)

        kld = gaussian_kld(mu, logsigma)
        self.avg_kld = torch.mean(kld)
        self.kl_weights = min(global_t / self.full_kl_step, 1.0)  # 退火
        self.kl_loss = self.kl_weights * self.avg_kld

        self.bow_logits = self.layers["bow_project_{}".format(sent_name)](z_post)
        labels = target[:, 1:]
        label_mask = torch.sign(labels).detach().float()
        bow_loss = -F.log_softmax(self.bow_logits, dim=1).gather(1, labels) * label_mask
        sum_bow_loss = torch.sum(bow_loss, 1)
        self.avg_bow_loss = torch.mean(sum_bow_loss)

        self.aug_elbo_loss = self.avg_bow_loss + self.kl_loss + self.rc_loss

        self.aug_elbo_loss.backward()
        self.optimizer_VAE[sent_name].step()

        avg_aug_elbo_loss = self.aug_elbo_loss.item()
        avg_rc_loss = self.rc_loss.data.item()
        avg_kl_loss = self.kl_loss.item()
        avg_bow_loss = self.avg_bow_loss.item()
        global_t += 1

        return [('avg_aug_elbo_loss', avg_aug_elbo_loss),
                ('avg_kl_loss', avg_kl_loss),
                ('avg_rc_loss', avg_rc_loss),
                ('avg_bow_loss', avg_bow_loss),
                ('kl_weight', self.kl_weights)], global_t

    def valid_VAE(self, global_t, target, target_lens, sent_name):
        self.eval()

        target_hidden, _ = self.layers["seq_encoder"](target[:, 1:], target_lens - 1)
        mu, logsigma, z_post = self.layers["{}_vae".format(sent_name)](target_hidden)
        output = self.layers["vae_decoder_{}".format(sent_name)](init_hidden=self.layers["init_decoder_hidden"](z_post), context=None, inputs=target[:, :-1])

        flattened_output = output.view(-1, self.vocab_size)
        dec_target = target[:, 1:].contiguous().view(-1)

        mask = dec_target.gt(0)  # 即判断target的token中是否有0（pad项）
        masked_target = dec_target.masked_select(mask)  # 选出非pad项
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)  # [(batch_sz * seq_len) x n_tokens]
        masked_output = flattened_output.masked_select(output_mask).view(-1, self.vocab_size)
        self.rc_loss = self.criterion_ce(masked_output / self.temp, masked_target)

        kld = gaussian_kld(mu, logsigma)
        self.avg_kld = torch.mean(kld)
        self.kl_weights = min(global_t / self.full_kl_step, 1.0)  # 退火
        self.kl_loss = self.kl_weights * self.avg_kld

        self.bow_logits = self.layers["bow_project_{}".format(sent_name)](z_post)
        labels = target[:, 1:]
        label_mask = torch.sign(labels).detach().float()
        bow_loss = -F.log_softmax(self.bow_logits, dim=1).gather(1, labels) * label_mask
        sum_bow_loss = torch.sum(bow_loss, 1)
        self.avg_bow_loss = torch.mean(sum_bow_loss)

        self.aug_elbo_loss = self.avg_bow_loss + self.kl_loss + self.rc_loss

        avg_aug_elbo_loss = self.aug_elbo_loss.item()
        avg_rc_loss = self.rc_loss.data.item()
        avg_kl_loss = self.kl_loss.item()
        avg_bow_loss = self.avg_bow_loss.item()

        return [('val_aug_elbo_loss', avg_aug_elbo_loss),
                ('val_kl_loss', avg_kl_loss),
                ('val_rc_loss', avg_rc_loss),
                ('val_bow_loss', avg_bow_loss),
                ('kl_weight', self.kl_weights)], global_t

    def test_VAE(self, sent_name, batch_size=1):
        z = to_tensor(torch.randn([batch_size, self.z_size]))
        pred_tokens = self.layers["vae_decoder_{}".format(sent_name)].testing(
            init_hidden=self.layers["init_decoder_hidden"](z),
            maxlen=self.maxlen,
            go_id=self.go_id,
            mode="greedy")
        pred_words = []
        # import pdb
        # pdb.set_trace()
        for b_id in range(pred_tokens.shape[0]):
            pred_words.append([self.vocab[e] for e in pred_tokens[b_id][:-1] if e != self.eos_id and e != 0 and e != self.go_id])
        return pred_words

    def train_AE(self, global_t, title, context, target, target_lens, sent_name):
        self.train()
        self.optimizer_AE[sent_name].zero_grad()
        # batch_size = title.size(0)
        # 每一句的情感用第二个分类器来预测，输入当前的m_hidden，输出分类结果
        title_last_hidden, _ = self.layers["seq_encoder"](title)
        context_last_hidden, _ = self.layers["seq_encoder"](context)
        target_hidden, _ = self.layers["seq_encoder"](target[:, 1:], target_lens - 1)

        mu, logsigma, z_post = self.layers["{}_vae".format(sent_name)](target_hidden)
        final_info = torch.cat([title_last_hidden, context_last_hidden, z_post], dim=1)

        output = self.layers["vae_decoder_{}".format(sent_name)](init_hidden=self.layers["init_decoder"](final_info),
                                                                 context=None, inputs=target[:, :-1])
        flattened_output = output.view(-1, self.vocab_size)
        dec_target = target[:, 1:].contiguous().view(-1)

        mask = dec_target.gt(0)  # 即判断target的token中是否有0（pad项）
        masked_target = dec_target.masked_select(mask)  # 选出非pad项
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)  # [(batch_sz * seq_len) x n_tokens]
        masked_output = flattened_output.masked_select(output_mask).view(-1, self.vocab_size)
        self.rc_loss = self.criterion_ce(masked_output / self.temp, masked_target)
        kld = gaussian_kld(mu, logsigma)
        self.avg_kld = torch.mean(kld)
        self.kl_weights = min(global_t / self.full_kl_step, 1.0)  # 退火
        self.kl_loss = self.kl_weights * self.avg_kld

        self.bow_logits = self.layers["bow_project_{}".format(sent_name)](z_post)
        labels = target[:, 1:]
        label_mask = torch.sign(labels).detach().float()
        bow_loss = -F.log_softmax(self.bow_logits, dim=1).gather(1, labels) * label_mask
        sum_bow_loss = torch.sum(bow_loss, 1)
        self.avg_bow_loss = torch.mean(sum_bow_loss)

        self.aug_elbo_loss = self.avg_bow_loss + self.kl_loss + self.rc_loss

        self.aug_elbo_loss.backward()

        self.optimizer_AE[sent_name].step()

        avg_aug_elbo_loss = self.aug_elbo_loss.item()
        avg_rc_loss = self.rc_loss.data.item()
        avg_kl_loss = self.kl_loss.item()
        avg_bow_loss = self.avg_bow_loss.item()
        global_t += 1

        return [('avg_aug_elbo_loss', avg_aug_elbo_loss),
                ('avg_kl_loss', avg_kl_loss),
                ('avg_rc_loss', avg_rc_loss),
                ('avg_bow_loss', avg_bow_loss),
                ('kl_weight', self.kl_weights)], global_t

    def valid_AE(self, global_t, title, context, target, target_lens, sent_name):
        self.eval()
        # batch_size = title.size(0)
        # 每一句的情感用第二个分类器来预测，输入当前的m_hidden，输出分类结果
        title_last_hidden, _ = self.layers["seq_encoder"](title)
        context_last_hidden, _ = self.layers["seq_encoder"](context)
        target_hidden, _ = self.layers["seq_encoder"](target[:, 1:], target_lens - 1)

        mu, logsigma, z_post = self.layers["{}_vae".format(sent_name)](target_hidden)
        final_info = torch.cat([title_last_hidden, context_last_hidden, z_post], dim=1)

        output = self.layers["vae_decoder_{}".format(sent_name)](init_hidden=self.layers["init_decoder"](final_info),
                                                                 context=None, inputs=target[:, :-1])
        flattened_output = output.view(-1, self.vocab_size)
        dec_target = target[:, 1:].contiguous().view(-1)

        mask = dec_target.gt(0)  # 即判断target的token中是否有0（pad项）
        masked_target = dec_target.masked_select(mask)  # 选出非pad项
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)  # [(batch_sz * seq_len) x n_tokens]
        masked_output = flattened_output.masked_select(output_mask).view(-1, self.vocab_size)
        self.rc_loss = self.criterion_ce(masked_output / self.temp, masked_target)
        kld = gaussian_kld(mu, logsigma)
        self.avg_kld = torch.mean(kld)
        self.kl_weights = min(global_t / self.full_kl_step, 1.0)  # 退火
        self.kl_loss = self.kl_weights * self.avg_kld

        self.bow_logits = self.layers["bow_project_{}".format(sent_name)](z_post)
        labels = target[:, 1:]
        label_mask = torch.sign(labels).detach().float()
        bow_loss = -F.log_softmax(self.bow_logits, dim=1).gather(1, labels) * label_mask
        sum_bow_loss = torch.sum(bow_loss, 1)
        self.avg_bow_loss = torch.mean(sum_bow_loss)

        self.aug_elbo_loss = self.avg_bow_loss + self.kl_loss + self.rc_loss

        avg_aug_elbo_loss = self.aug_elbo_loss.item()
        avg_rc_loss = self.rc_loss.data.item()
        avg_kl_loss = self.kl_loss.item()
        avg_bow_loss = self.avg_bow_loss.item()

        return [('valid_aug_elbo_loss', avg_aug_elbo_loss),
                ('valid_kl_loss', avg_kl_loss),
                ('valid_rc_loss', avg_rc_loss),
                ('valid_bow_loss', avg_bow_loss)]

    # batch_size = 1 只输入了一个标题
    # test的时候，只有先验，没有后验，更没有所谓的kl散度
    def test(self, title_tensor, title_words, sent_labels):
        self.eval()
        name_dict = {'0': 'neg', '1': 'neu', '2': 'pos'}
        batch_size = title_tensor.size(0)
        assert batch_size == 1
        tem = [[2, 3] + [0] * (self.maxlen - 2)]
        pred_poems = []
        # 过滤掉标题中的<s> </s> 0,只为了打印
        title_tokens = [self.vocab[e] for e in title_words[0].tolist() if e not in [0, self.eos_id, self.go_id]]
        pred_poems.append(title_tokens)

        gen_words = ""
        gen_tokens = []

        for i in range(4):
            tem = to_tensor(np.array(tem))
            context = tem
            if i == 0:
                context_last_hidden, _ = self.layers["seq_encoder"](title_tensor)
            else:
                context_last_hidden, _ = self.layers["seq_encoder"](context)
            title_last_hidden, _ = self.layers["seq_encoder"](title_tensor)
            z = to_tensor(torch.randn([batch_size, self.z_size]))

            final_info = torch.cat([title_last_hidden, context_last_hidden, z], dim=1)
            pred_tokens = self.layers["vae_decoder_{}".format(name_dict[sent_labels[i]])].testing(init_hidden=self.layers["init_decoder"](final_info), maxlen=self.maxlen, go_id=self.go_id, mode="greedy")
            pred_tokens = pred_tokens[0].tolist()

            if len(pred_tokens) >= self.maxlen:
                tem = [pred_tokens[0: self.maxlen]]
            else:
                tem = [[0] * (self.maxlen - len(pred_tokens)) + pred_tokens]

            pred_words = [self.vocab[e] for e in pred_tokens[:-1] if e != self.eos_id and e != 0 and e != self.go_id]
            pred_poems.append(pred_words)
            gen_tokens.append(pred_tokens)

        for i in range(5):
            if i == 0:
                cur_line = " ".join(pred_poems[i])
            else:
                cur_line = " ".join(pred_poems[i]) + sent_labels[i-1]
            gen_words = gen_words + cur_line + '\n'

        return gen_words, gen_tokens

    def sample(self, title, context, repeat, go_id, end_id):
        self.seq_encoder.eval()
        self.decoder.eval()

        assert title.size(0) == 1
        title_last_hidden, _ = self.seq_encoder(title)
        context_last_hidden, _ = self.seq_encoder(context)
        condition_prior = torch.cat((title_last_hidden, context_last_hidden), 1)
        condition_prior_repeat = condition_prior.expand(repeat, -1)

        # z_prior_repeat, _, _, _, _ = self.sample_code_prior(condition_prior_repeat)
        z_prior_repeat, _, _ = self.sample_code_prior(condition_prior_repeat)

        final_info = torch.cat((z_prior_repeat, condition_prior_repeat), dim=1)
        sample_words, sample_lens = self.decoder.sampling(init_hidden=self.init_decoder_hidden(final_info), maxlen=self.maxlen,
                                                          go_id=self.go_id, eos_id=self.eos_id, mode="greedy")

        return sample_words, sample_lens
