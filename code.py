# paste me to modelling_mmada.py to replace t2i_generate() and try it!
    @torch.no_grad()
    def t2i_generate_reddit(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            uncond_attention_mask=None,
            temperature=1.0,
            timesteps=18,  
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            seq_len=1024,
            mask_token_id = 126336,
            resolution = 512,
            codebook_size = 8192,
            **kwargs,
    ):
        """
        A low-discrepancy sampler modified by ReDDiT.
        """

        # begin with all image token ids masked
        num_vq_tokens = seq_len
        num_new_special_tokens = 0
        uni_prompting = kwargs.get("uni_prompting", None)
        vocab_shift = len(uni_prompting.text_tokenizer) + num_new_special_tokens
        # print(f"config.model.mmada.llm_vocab_size: {config.model.mmada.llm_vocab_size}, {len(uni_prompting.text_tokenizer)}")
        sampled_ids = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        # this results in the uni-mask canvas

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :resolution + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, resolution + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(model_input, attention_bias=attention_bias).logits 
                # print(f"logits.shape: {logits.shape}")
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, vocab_shift: vocab_shift + codebook_size]
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(input_ids, attention_bias=attention_bias).logits
                logits = logits[:, -(num_vq_tokens + 1):-1, vocab_shift: vocab_shift + codebook_size]

            # logits: 1, 1024, 8192
            # print(f"logits.shape: {logits.shape}")
            probs = logits.softmax(dim=-1)
            if step < timesteps - 1:
                k_t = noise_schedule(torch.tensor(1.0 * (step) / timesteps))
                k_s = noise_schedule(torch.tensor(1.0 * (step + 1) / timesteps))
                p_mask = (k_s / k_t).expand(probs.shape[0], probs.shape[1], 1).to(probs.device) # for uni-mask tokens

                probs_with_mask_index = torch.cat([probs * (k_t - k_s) / k_t, p_mask], dim=-1)
                new_pred_with_masks = sample_categorical(probs_with_mask_index)

                updated_sampled_ids = torch.where(sampled_ids > codebook_size -1, new_pred_with_masks, sampled_ids)
                sampled_ids = updated_sampled_ids
            else:
                new_pred = sample_categorical(probs)
                sampled_ids = torch.where(sampled_ids > codebook_size -1, new_pred, sampled_ids)
            # print(f'decoded tokens:{(sampled_ids < codebook_size).sum().item()}')
            sampled_ids_to_paste = torch.where(sampled_ids > codebook_size -1, mask_token_id - vocab_shift, updated_sampled_ids)
            input_ids[:, -(num_vq_tokens + 1):-1] = sampled_ids_to_paste + vocab_shift
        return sampled_ids