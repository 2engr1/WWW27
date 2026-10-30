"""Token-level sufficiency, necessity, and bidirectional NS constraints."""
from __future__ import annotations
from models import RetrievedDoc, harmonic_mean_score

class BidirectionalTokenMatcher:
    """NS-Retrieval's first-stage token-level bidirectional constraint.

    The existing SentenceTransformer's underlying Transformer and tokenizer are
    reused; no new encoder is trained and no vector index is changed. Document
    token sequences longer than the encoder limit are processed in non-overlapping
    windows so that every content token contributes once to the document-side mean.
    """

    def __init__(self, sentence_transformer, batch_size: int = 16, score_mode: str = "ns"):
        import torch

        if score_mode not in {"ns", "suf", "nec"}:
            raise ValueError("score_mode must be one of: 'ns', 'suf', 'nec'")
        self.score_mode = score_mode
        self.score_source = {
            "ns": "ns-bidirectional",
            "suf": "ns-suf",
            "nec": "ns-nec",
        }[score_mode]
        self.torch = torch
        self.batch_size = max(1, int(batch_size))
        try:
            transformer_module = sentence_transformer._first_module()
        except AttributeError:
            transformer_module = sentence_transformer[0]
        self.tokenizer = transformer_module.tokenizer
        self.transformer = transformer_module.auto_model
        self.device = next(self.transformer.parameters()).device
        self.max_length = int(getattr(transformer_module, "max_seq_length", 512))
        special_counter = getattr(self.tokenizer, "num_special_tokens_to_add", None)
        try:
            special_count = int(special_counter(pair=False)) if callable(special_counter) else 2
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            # E5/BERT single-sequence inputs use CLS and SEP; otherwise prefer
            # whatever special-token identifiers the tokenizer exposes.
            cls_id = getattr(self.tokenizer, "cls_token_id", None)
            sep_id = getattr(self.tokenizer, "sep_token_id", None)
            bos_id = getattr(self.tokenizer, "bos_token_id", None)
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            special_count = 2 if (cls_id is not None and sep_id is not None) or (
                bos_id is not None and eos_id is not None
            ) else int(eos_id is not None or sep_id is not None)
        self.max_content_tokens = max(1, self.max_length - special_count)
        self.transformer.eval()

    @staticmethod
    def _as_list(value):
        """Normalize tokenizer outputs (lists, tensors, or one-item batches)."""
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
            value = list(value[0])
        return list(value) if value is not None else []

    def _infer_special_mask(self, input_ids: list[int], content_ids: list[int]) -> list[int] | None:
        """Infer special-token positions by locating the no-special content sequence."""
        if not content_ids:
            return [1] * len(input_ids)
        width = len(content_ids)
        for start in range(0, len(input_ids) - width + 1):
            if input_ids[start:start + width] == content_ids:
                mask = [1] * len(input_ids)
                mask[start:start + width] = [0] * width
                return mask
        return None

    def _prepare(self, token_ids: list[int]) -> dict:
        """Prepare a single sequence across Transformers/tokenizer API variants.

        Use the fast public helpers when present. If a tokenizer wrapper omits both
        preparation helpers and token ID attributes (seen in some deployed
        BertTokenizer environments), round-trip through ``decode`` and the
        tokenizer's normal callable interface. This is the same input route used
        by SentenceTransformer.encode(), so it does not require changing packages.
        """
        token_ids = [int(value) for value in token_ids]
        prepare = getattr(self.tokenizer, "prepare_for_model", None)
        if callable(prepare):
            try:
                prepared = dict(prepare(
                    token_ids,
                    add_special_tokens=True,
                    return_attention_mask=True,
                    return_special_tokens_mask=True,
                    truncation=False,
                ))
                if "input_ids" in prepared:
                    input_ids = self._as_list(prepared["input_ids"])
                    attention_mask = self._as_list(prepared.get("attention_mask")) or [1] * len(input_ids)
                    special_mask = self._as_list(prepared.get("special_tokens_mask"))
                    if len(special_mask) != len(input_ids):
                        special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
                        special_mask = ([int(value in special_ids) for value in input_ids]
                                        if special_ids else self._infer_special_mask(input_ids, token_ids))
                    if special_mask is None or len(special_mask) != len(input_ids):
                        raise ValueError("Cannot identify special-token positions from prepared input")
                    result = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "special_tokens_mask": special_mask,
                    }
                    token_type_ids = self._as_list(prepared.get("token_type_ids"))
                    if len(token_type_ids) == len(input_ids):
                        result["token_type_ids"] = token_type_ids
                    return result
            except (AttributeError, NotImplementedError, TypeError, ValueError):
                pass

        input_ids = None
        special_mask = None
        token_type_ids = None
        content_ids = token_ids

        builder = getattr(self.tokenizer, "build_inputs_with_special_tokens", None)
        if callable(builder):
            try:
                input_ids = self._as_list(builder(token_ids))
                mask_builder = getattr(self.tokenizer, "get_special_tokens_mask", None)
                if callable(mask_builder):
                    try:
                        special_mask = self._as_list(mask_builder(
                            token_ids, already_has_special_tokens=False
                        ))
                    except (AttributeError, NotImplementedError, TypeError, ValueError):
                        special_mask = None
            except (AttributeError, NotImplementedError, TypeError, ValueError):
                input_ids = None
                special_mask = None

        # Some wrappers expose neither helper but still expose special-token IDs.
        if input_ids is None:
            cls_id = getattr(self.tokenizer, "cls_token_id", None)
            sep_id = getattr(self.tokenizer, "sep_token_id", None)
            bos_id = getattr(self.tokenizer, "bos_token_id", None)
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            if cls_id is not None and sep_id is not None:
                input_ids = [int(cls_id), *token_ids, int(sep_id)]
                special_mask = [1, *([0] * len(token_ids)), 1]
            elif bos_id is not None and eos_id is not None:
                input_ids = [int(bos_id), *token_ids, int(eos_id)]
                special_mask = [1, *([0] * len(token_ids)), 1]
            elif eos_id is not None:
                input_ids = [*token_ids, int(eos_id)]
                special_mask = [*([0] * len(token_ids)), 1]
            elif sep_id is not None:
                input_ids = [*token_ids, int(sep_id)]
                special_mask = [*([0] * len(token_ids)), 1]

        token_type_ids = None
        if input_ids is None:
            # Crucial compatibility path: use the tokenizer call interface rather
            # than requiring prepare_for_model/build_inputs_with_special_tokens.
            decoder = getattr(self.tokenizer, "decode", None)
            if callable(decoder):
                try:
                    text = decoder(
                        token_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                except TypeError:
                    text = decoder(token_ids, skip_special_tokens=True)
                try:
                    encoded = self.tokenizer(
                        text,
                        add_special_tokens=True,
                        return_attention_mask=True,
                        return_special_tokens_mask=True,
                        truncation=False,
                    )
                except (AttributeError, NotImplementedError, TypeError, ValueError):
                    try:
                        encoded = self.tokenizer(
                            text,
                            add_special_tokens=True,
                            return_attention_mask=True,
                            truncation=False,
                        )
                    except (AttributeError, NotImplementedError, TypeError, ValueError):
                        encoded = None
                if encoded is not None:
                    try:
                        input_ids = self._as_list(encoded.get("input_ids"))
                        special_mask = self._as_list(encoded.get("special_tokens_mask"))
                        token_type_ids = self._as_list(encoded.get("token_type_ids"))
                    except AttributeError:
                        input_ids = []
                        special_mask = []
                        token_type_ids = []
                content_ids = self._as_list(
                    self.tokenizer.encode(text, add_special_tokens=False)
                )

            # A final encode-based path covers callable wrappers that do not return
            # a mapping with input_ids. We still infer the content span explicitly.
            if not input_ids and callable(decoder):
                input_ids = self._as_list(
                    self.tokenizer.encode(text, add_special_tokens=True)
                )

        if not input_ids:
            raise RuntimeError(
                f"Tokenizer {type(self.tokenizer).__name__} cannot prepare token-level NS input: "
                "preparation helpers, special-token IDs, and callable/encode fallback are unavailable."
            )

        if not special_mask or len(special_mask) != len(input_ids):
            special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
            if special_ids:
                special_mask = [int(token_id in special_ids) for token_id in input_ids]
            else:
                special_mask = self._infer_special_mask(input_ids, content_ids)
        if special_mask is None or len(special_mask) != len(input_ids):
            raise RuntimeError(
                f"Could not identify special-token positions for {type(self.tokenizer).__name__}; "
                "refusing to score boundary tokens as evidence."
            )

        feature = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "special_tokens_mask": special_mask,
        }
        if token_type_ids and len(token_type_ids) == len(input_ids):
            feature["token_type_ids"] = token_type_ids
        return feature

    def _pad_feature_batch(self, group: list[dict]) -> dict:
        """Pad tokenizer features with PyTorch, independent of tokenizer.pad APIs."""
        torch = self.torch
        if not group:
            return {}
        lengths = [len(feature["input_ids"]) for feature in group]
        max_length = max(lengths, default=0)
        # If a tokenizer has no declared pad ID, zero is safe for masked padding:
        # attention_mask=0 ensures its hidden state is never scored as content.
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        pad_id = 0 if pad_id is None else int(pad_id)
        padding_side = getattr(self.tokenizer, "padding_side", "right")

        input_ids = torch.full((len(group), max_length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(group), max_length), dtype=torch.long)
        # Mark padding as special as an additional guard against scoring pad tokens.
        special_tokens_mask = torch.ones((len(group), max_length), dtype=torch.long)
        has_token_type_ids = any("token_type_ids" in feature for feature in group)
        token_type_ids = (torch.zeros((len(group), max_length), dtype=torch.long)
                          if has_token_type_ids else None)

        for row, feature in enumerate(group):
            ids = list(feature["input_ids"])
            length = len(ids)
            offset = max_length - length if padding_side == "left" else 0
            if length:
                input_ids[row, offset:offset + length] = torch.as_tensor(ids, dtype=torch.long)
                attn = feature.get("attention_mask", [1] * length)
                mask = feature.get("special_tokens_mask", [0] * length)
                attention_mask[row, offset:offset + length] = torch.as_tensor(attn, dtype=torch.long)
                special_tokens_mask[row, offset:offset + length] = torch.as_tensor(mask, dtype=torch.long)
                if token_type_ids is not None:
                    types = feature.get("token_type_ids", [0] * length)
                    token_type_ids[row, offset:offset + length] = torch.as_tensor(types, dtype=torch.long)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": special_tokens_mask,
        }
        if token_type_ids is not None:
            batch["token_type_ids"] = token_type_ids
        return batch

    def _iter_encoded_tokens(self, features: list[dict]):
        """Yield (feature index, normalized content-token vectors) in batches."""
        torch = self.torch
        for start in range(0, len(features), self.batch_size):
            group = features[start : start + self.batch_size]
            batch = self._pad_feature_batch(group)
            batch = {key: value.to(self.device) for key, value in batch.items()}
            attention = batch.get("attention_mask")
            special_mask = batch.pop("special_tokens_mask", None)
            model_inputs = {
                key: value
                for key, value in batch.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }
            with torch.inference_mode():
                output = self.transformer(**model_inputs)
                hidden = torch.nn.functional.normalize(output.last_hidden_state, p=2, dim=-1)
            if attention is None:
                valid = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
            else:
                valid = attention.bool()
            if special_mask is not None:
                valid = valid & (~special_mask.bool())
            for local_index in range(len(group)):
                vectors = hidden[local_index][valid[local_index]]
                yield start + local_index, vectors

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedDoc],
        top_k: int | None = None,
    ) -> list[RetrievedDoc]:
        if not candidates:
            return []
        torch = self.torch
        query_ids = self.tokenizer.encode(query, add_special_tokens=False)
        query_ids = query_ids[: self.max_content_tokens]
        if not query_ids:
            return [
                RetrievedDoc(d.doc_id, d.text, 0.0, self.score_source, d.metadata)
                for d in candidates[:top_k]
            ]
        query_features = [self._prepare(query_ids)]
        query_tokens = next(self._iter_encoded_tokens(query_features))[1]
        if query_tokens.numel() == 0:
            return [
                RetrievedDoc(d.doc_id, d.text, 0.0, self.score_source, d.metadata)
                for d in candidates[:top_k]
            ]

        # Build a list of document windows while retaining their owning candidate.
        window_features: list[dict] = []
        window_owners: list[int] = []
        for doc_index, doc in enumerate(candidates):
            token_ids = self.tokenizer.encode(doc.text or "", add_special_tokens=False)
            if not token_ids:
                continue
            for start in range(0, len(token_ids), self.max_content_tokens):
                segment = token_ids[start : start + self.max_content_tokens]
                if segment:
                    window_features.append(self._prepare(segment))
                    window_owners.append(doc_index)

        query_maxima = [torch.zeros(query_tokens.shape[0], device=self.device) for _ in candidates]
        document_sums = [0.0 for _ in candidates]
        document_counts = [0 for _ in candidates]
        for feature_index, document_tokens in self._iter_encoded_tokens(window_features):
            if document_tokens.numel() == 0:
                continue
            owner = window_owners[feature_index]
            # M_ij = max(0, cosine(x_i, y_j)); vectors are L2-normalized.
            matching = torch.matmul(document_tokens, query_tokens.transpose(0, 1)).clamp(min=0.0)
            if self.score_mode in {"ns", "suf"}:
                query_maxima[owner] = torch.maximum(query_maxima[owner], matching.max(dim=0).values)
            if self.score_mode in {"ns", "nec"}:
                document_sums[owner] += float(matching.max(dim=1).values.sum().item())
            document_counts[owner] += int(document_tokens.shape[0])

        scored = []
        for index, doc in enumerate(candidates):
            if document_counts[index] == 0:
                suf = 0.0
                nec = 0.0
            else:
                # Query-side coverage and document-side matching, respectively.
                suf = float(query_maxima[index].mean().item())
                nec = document_sums[index] / document_counts[index]
            if self.score_mode == "suf":
                score = suf
            elif self.score_mode == "nec":
                score = nec
            else:
                score = harmonic_mean_score(suf, nec)
            scored.append(RetrievedDoc(
                doc.doc_id, doc.text, score, self.score_source, doc.metadata,
                score_components={
                    "sufficiency": suf if self.score_mode in {"ns", "suf"} else None,
                    "necessity": nec if self.score_mode in {"ns", "nec"} else None,
                    "bidirectional": harmonic_mean_score(suf, nec) if self.score_mode == "ns" else None,
                },
            ))
        scored.sort(key=lambda doc: -doc.score)
        return scored if top_k is None else scored[:top_k]
