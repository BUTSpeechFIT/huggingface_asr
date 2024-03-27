# pylint: skip-file
# Copied from: https://github.com/espnet/espnet/blob/master/espnet/nets/ctc_prefix_score.py
import torch
from transformers import LogitsProcessor


class CTCPrefixScoreTH(object):
    """Batch processing of CTCPrefixScore

    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the label probablities for multiple
    hypotheses simultaneously
    See also Seki et al. "Vectorized Beam Search for CTC-Attention-Based
    Speech Recognition," In INTERSPEECH (pp. 3825-3829), 2019.
    """

    def __init__(self, x, xlens, blank, eos, margin=0):
        """Construct CTC prefix scorer

        :param torch.Tensor x: input label posterior sequences (B, T, O)
        :param torch.Tensor xlens: input lengths (B,)
        :param int blank: blank label id
        :param int eos: end-of-sequence id
        :param int margin: margin parameter for windowing (0 means no windowing)
        """
        # In the comment lines,
        # we assume T: input_length, B: batch size, W: beam width, O: output dim.
        self.logzero = -10000000000.0
        self.blank = blank
        self.eos = eos
        self.batch = x.size(0)
        self.input_length = x.size(1)
        self.odim = x.size(2)
        self.dtype = x.dtype
        self.device = torch.device("cuda:%d" % x.get_device()) if x.is_cuda else torch.device("cpu")
        # Pad the rest of posteriors in the batch
        # TODO(takaaki-hori): need a better way without for-loops
        for i, l in enumerate(xlens):
            if l < self.input_length:
                x[i, l:, :] = self.logzero
                x[i, l:, blank] = 0
        # Reshape input x
        xn = x.transpose(0, 1)  # (B, T, O) -> (T, B, O)
        xb = xn[:, :, self.blank].unsqueeze(2).expand(-1, -1, self.odim)
        self.x = torch.stack([xn, xb])  # (2, T, B, O)
        self.end_frames = torch.as_tensor(xlens) - 1

        # Setup CTC windowing
        self.margin = margin
        if margin > 0:
            self.frame_ids = torch.arange(self.input_length, dtype=self.dtype, device=self.device)
        # Base indices for index conversion
        self.idx_bh = None
        self.idx_b = torch.arange(self.batch, device=self.device)
        self.idx_bo = (self.idx_b * self.odim).unsqueeze(1)

    def __call__(self, y, state, scoring_ids=None, att_w=None):
        """Compute CTC prefix scores for next labels

        :param list y: prefix label sequences
        :param tuple state: previous CTC state
        :param torch.Tensor att_w: attention weights to decide CTC window
        :return new_state, ctc_local_scores (BW, O)
        """

        # print(self.tokenizer.batch_decode(y))
        output_length = len(y[0]) - 1  # ignore sos
        last_ids = [yi[-1] for yi in y]  # last output label ids
        n_bh = len(last_ids)  # batch * hyps
        n_hyps = n_bh // self.batch  # assuming each utterance has the same # of hyps
        self.scoring_num = scoring_ids.size(-1) if scoring_ids is not None else 0
        # prepare state info
        if state is None:
            r_prev = torch.full(
                (self.input_length, 2, self.batch, n_hyps),
                self.logzero,
                dtype=self.dtype,
                device=self.device,
            )
            r_prev[:, 1] = torch.cumsum(self.x[0, :, :, self.blank], 0).unsqueeze(2)
            r_prev = r_prev.view(-1, 2, n_bh)
            s_prev = 0.0
            f_min_prev = 0
            f_max_prev = 1
        else:
            r_prev, s_prev, f_min_prev, f_max_prev = state

        # select input dimensions for decred_scoring
        if self.scoring_num > 0:
            scoring_idmap = torch.full((n_bh, self.odim), -1, dtype=torch.long, device=self.device)
            snum = self.scoring_num
            if self.idx_bh is None or n_bh > len(self.idx_bh):
                self.idx_bh = torch.arange(n_bh, device=self.device).view(-1, 1)
            scoring_idmap[self.idx_bh[:n_bh], scoring_ids] = torch.arange(snum, device=self.device)
            scoring_idx = (scoring_ids + self.idx_bo.repeat(1, n_hyps).view(-1, 1)).view(-1)
            x_ = torch.index_select(self.x.view(2, -1, self.batch * self.odim), 2, scoring_idx).view(2, -1, n_bh, snum)
        else:
            scoring_ids = None
            scoring_idmap = None
            snum = self.odim
            x_ = self.x.unsqueeze(3).repeat(1, 1, 1, n_hyps, 1).view(2, -1, n_bh, snum)

        # new CTC forward probs are prepared as a (T x 2 x BW x S) tensor
        # that corresponds to r_t^n(h) and r_t^b(h) in a batch.
        r = torch.full(
            (self.input_length, 2, n_bh, snum),
            self.logzero,
            dtype=self.dtype,
            device=self.device,
        )
        if output_length == 0:
            r[0, 0] = x_[0, 0]

        r_sum = torch.logsumexp(r_prev, 1)
        log_phi = r_sum.unsqueeze(2).repeat(1, 1, snum)
        if scoring_ids is not None:
            for idx in range(n_bh):
                pos = scoring_idmap[idx, last_ids[idx]]
                if pos >= 0:
                    log_phi[:, idx, pos] = r_prev[:, 1, idx]
        else:
            for idx in range(n_bh):
                log_phi[:, idx, last_ids[idx]] = r_prev[:, 1, idx]

        # decide start and end frames based on attention weights
        if att_w is not None and self.margin > 0:
            f_arg = torch.matmul(att_w, self.frame_ids)
            f_min = max(int(f_arg.min().cpu()), f_min_prev)
            f_max = max(int(f_arg.max().cpu()), f_max_prev)
            start = min(f_max_prev, max(f_min - self.margin, output_length, 1))
            end = min(f_max + self.margin, self.input_length)
        else:
            f_min = f_max = 0
            start = max(output_length, 1)
            end = self.input_length

        if start > end:
            return torch.full_like(s_prev, self.logzero), (
                r,
                torch.full_like(s_prev, self.logzero),
                f_min,
                f_max,
                scoring_idmap,
            )

        # compute forward probabilities log(r_t^n(h)) and log(r_t^b(h))
        for t in range(start, end):
            rp = r[t - 1]
            rr = torch.stack([rp[0], log_phi[t - 1], rp[0], rp[1]]).view(2, 2, n_bh, snum)
            r[t] = torch.logsumexp(rr, 1) + x_[:, t]

        # compute log prefix probabilities log(psi)
        log_phi_x = torch.cat((log_phi[0].unsqueeze(0), log_phi[:-1]), dim=0) + x_[0]
        if scoring_ids is not None:
            log_psi = torch.full((n_bh, self.odim), self.logzero, dtype=self.dtype, device=self.device)
            log_psi_ = torch.logsumexp(
                torch.cat((log_phi_x[start:end], r[start - 1, 0].unsqueeze(0)), dim=0),
                dim=0,
            )
            for si in range(n_bh):
                log_psi[si, scoring_ids[si]] = log_psi_[si]
        else:
            log_psi = torch.logsumexp(
                torch.cat((log_phi_x[start:end], r[start - 1, 0].unsqueeze(0)), dim=0),
                dim=0,
            )

        # for si in range(n_bh):
        #     log_psi[si, self.eos] = r_sum[self.end_frames[si // n_hyps], si]

        # exclude blank probs
        log_psi[:, self.blank] = self.logzero

        token_scores = log_psi - s_prev
        token_scores[token_scores == 0] = self.logzero

        return token_scores, (r, log_psi, f_min, f_max, scoring_idmap)

    def index_select_state(self, state, best_ids):
        """Select CTC states according to best ids

        :param state    : CTC state
        :param best_ids : index numbers selected by beam pruning (B, W)
        :return selected_state
        """
        r, s, f_min, f_max, scoring_idmap = state
        # convert ids to BHO space
        n_bh = len(s)
        n_hyps = n_bh // self.batch
        vidx = (best_ids + (self.idx_b * (n_hyps * self.odim)).view(-1, 1)).view(-1)
        # select hypothesis scores
        s_new = torch.index_select(s.view(-1), 0, vidx)
        s_new = s_new.view(-1, 1).repeat(1, self.odim).view(n_bh, self.odim)
        # convert ids to BHS space (S: scoring_num)
        if scoring_idmap is not None:
            snum = self.scoring_num
            hyp_idx = (best_ids // self.odim + (self.idx_b * n_hyps).view(-1, 1)).view(-1)
            label_ids = torch.fmod(best_ids, self.odim).view(-1)
            score_idx = scoring_idmap[hyp_idx, label_ids]
            score_idx[score_idx == -1] = 0
            vidx = score_idx + hyp_idx * snum
        else:
            snum = self.odim
        # select forward probabilities
        r_new = torch.index_select(r.view(-1, 2, n_bh * snum), 2, vidx).view(-1, 2, n_bh)
        return r_new, s_new, f_min, f_max

    def extend_prob(self, x):
        """Extend CTC prob.

        :param torch.Tensor x: input label posterior sequences (B, T, O)
        """

        if self.x.shape[1] < x.shape[1]:  # self.x (2,T,B,O); x (B,T,O)
            # Pad the rest of posteriors in the batch
            # TODO(takaaki-hori): need a better way without for-loops
            xlens = [x.size(1)]
            for i, l in enumerate(xlens):
                if l < self.input_length:
                    x[i, l:, :] = self.logzero
                    x[i, l:, self.blank] = 0
            tmp_x = self.x
            xn = x.transpose(0, 1)  # (B, T, O) -> (T, B, O)
            xb = xn[:, :, self.blank].unsqueeze(2).expand(-1, -1, self.odim)
            self.x = torch.stack([xn, xb])  # (2, T, B, O)
            self.x[:, : tmp_x.shape[1], :, :] = tmp_x
            self.input_length = x.size(1)
            self.end_frames = torch.as_tensor(xlens) - 1

    def extend_state(self, state):
        """Compute CTC prefix state.


        :param state    : CTC state
        :return ctc_state
        """

        if state is None:
            # nothing to do
            return state
        else:
            r_prev, s_prev, f_min_prev, f_max_prev = state

            r_prev_new = torch.full(
                (self.input_length, 2),
                self.logzero,
                dtype=self.dtype,
                device=self.device,
            )
            start = max(r_prev.shape[0], 1)
            r_prev_new[0:start] = r_prev
            for t in range(start, self.input_length):
                r_prev_new[t, 1] = r_prev_new[t - 1, 1] + self.x[0, t, :, self.blank]

            return (r_prev_new, s_prev, f_min_prev, f_max_prev)


class CTCRescorerLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        encoder_logits: torch.FloatTensor,
        encoder_output_lens: torch.LongTensor,
        pad_token_id: int,
        eos_token_id: int,
        ctc_margin: int,
        ctc_weight: float,
        num_beams: int,
        space_token_id: int,
        apply_eos_space_trick: bool,
        eos_space_trick_weight: float,
        debug: bool = False,
    ):
        super().__init__()
        # reduce_lens_by = (encoder_logits.argmax(dim=-1) == eos_token_id).sum(dim=-1)
        # encoder_output_lens = encoder_output_lens - reduce_lens_by
        self.pad_token_id = pad_token_id
        self.ctc_prefix_scorer = CTCPrefixScoreTH(
            torch.nn.functional.log_softmax(encoder_logits, dim=-1),
            encoder_output_lens,
            pad_token_id,
            eos_token_id,
            ctc_margin,
        )
        self.ctc_weight = ctc_weight
        self.ctc_states = None
        self.num_beams = num_beams
        self.eos_token_id = eos_token_id
        self.apply_eos_space_trick = apply_eos_space_trick
        self.space_token_id = space_token_id
        self.eos_space_trick_weight = eos_space_trick_weight
        self.debug = debug

    @staticmethod
    def analyze_predictions(
        scores, ctc_scores, next_token_scores, input_ids, k=10, tokenizer="Lakoc/english_corpus_uni5000_normalized"
    ):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        best_att_ids = scores.topk(k=k, dim=1)
        best_ctc_ids = ctc_scores.topk(k=k, dim=1)
        best_ids = next_token_scores.topk(k=k, dim=1)

        def print_prediction(best_ids, name):
            new_tensor = torch.zeros((best_ids.indices.shape[0], best_ids.indices.shape[1] * 2), dtype=torch.long)
            new_tensor[:, 0::2] = best_ids.indices
            new_tensor[:, 1::2] = 4976
            print(f"{name}:")
            for index, (next_ids, scores) in enumerate(zip(tokenizer.batch_decode(new_tensor), best_ids.values)):
                print(f"HYP {index}:\n{next_ids} {scores}")

        print(f"PREFIX:")
        for index, prefix in enumerate(tokenizer.batch_decode(input_ids)):
            print(f"HYP {index}:\n{prefix}")
        print_prediction(best_att_ids, "ATT_SCORES")
        print()
        print_prediction(best_ctc_ids, "CTC_SCORES")
        print()
        print(f"CTC_EOS: {ctc_scores[:, 1]}")
        print_prediction(best_ids, "NEXT_TOKEN_SCORES")
        print()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self.pad_token_id] = self.ctc_prefix_scorer.logzero
        if self.ctc_states is not None:
            self.ctc_states = self.ctc_prefix_scorer.index_select_state(
                self.ctc_states, input_ids[:, -1].reshape(-1, self.num_beams)
            )
        ctc_scores, ctc_states = self.ctc_prefix_scorer(input_ids, self.ctc_states)
        self.ctc_states = ctc_states
        next_token_scores = (1 - self.ctc_weight) * scores + self.ctc_weight * ctc_scores
        if self.apply_eos_space_trick:
            space_eos_conflict = torch.logical_and(
                scores.argmax(dim=1) == self.eos_token_id, ctc_scores.argmax(dim=1) == self.space_token_id
            )
            if space_eos_conflict.any():
                apply_trick_on = torch.logical_and(
                    torch.logical_and(
                        space_eos_conflict,
                        next_token_scores[:, self.eos_token_id] < next_token_scores[:, self.space_token_id],
                    ),
                    self.eos_space_trick_weight * next_token_scores[:, self.eos_token_id]
                    > next_token_scores[:, self.space_token_id],
                )
                if apply_trick_on.any():
                    next_token_scores[apply_trick_on, self.eos_token_id] = (
                        next_token_scores[apply_trick_on, self.eos_token_id] * self.eos_space_trick_weight
                    )

        if self.debug:
            self.analyze_predictions(scores, ctc_scores, next_token_scores, input_ids)

        return next_token_scores


class LogSoftmaxProcessor(LogitsProcessor):
    def __init__(
        self,
    ):
        super().__init__()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        return scores
