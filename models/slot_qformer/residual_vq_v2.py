from vector_quantize_pytorch.residual_vq import *

# main class

class ResidualVQ(ResidualVQ):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """

    def __init__(
            self,
            *,
            dim,
            num_quantizers: int | None = None,
            codebook_size: int | tuple[int, ...],
            codebook_dim=None,
            shared_codebook=False,
            shared_second_codebook=False,  # add
            use_projection_bias=True,  # add
            heads=1,
            quantize_dropout=False,
            quantize_dropout_cutoff_index=0,
            quantize_dropout_multiple_of=1,
            accept_image_fmap=False,
            implicit_neural_codebook=False,  # QINCo from https://arxiv.org/abs/2401.14732
            mlp_kwargs: dict = dict(),
            **vq_kwargs
    ):
        super().__init__(
            dim=dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            shared_codebook=shared_codebook,
            heads=heads,
            quantize_dropout=quantize_dropout,
            quantize_dropout_cutoff_index=quantize_dropout_cutoff_index,
            quantize_dropout_multiple_of=quantize_dropout_multiple_of,
            accept_image_fmap=accept_image_fmap,
            implicit_neural_codebook=implicit_neural_codebook,
            mlp_kwargs=mlp_kwargs,
            **vq_kwargs,
        )

        assert heads == 1, 'residual vq is not compatible with multi-headed codes'
        assert exists(num_quantizers) or isinstance(codebook_size, tuple)

        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        if requires_projection:

            self.project_in = nn.Linear(dim, codebook_input_dim) if use_projection_bias else nn.Linear(dim,
                                                                                                       codebook_input_dim,
                                                                                                       bias=False)
            self.project_out = nn.Linear(codebook_input_dim, dim) if use_projection_bias else nn.Linear(
                codebook_input_dim, dim, bias=False)
        else:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()

        if not shared_second_codebook:

            first_vq, *rest_vq = self.layers
            codebook = first_vq._codebook

            for vq in rest_vq:
                vq._codebook = codebook
        else:
            first_vq, second_vq, *rest_vq = self.layers
            codebook = second_vq._codebook

            for vq in rest_vq:
                vq._codebook = codebook

    def forward(
            self,
            x,
            mask=None,
            indices: Tensor | list[Tensor] | None = None,
            return_all_codes=False,
            return_quantized_out=False,
            sample_codebook_temp=None,
            freeze_codebook=False,
            rand_quantize_dropout_fixed_seed=None
    ):
        num_quant, quant_dropout_multiple_of, return_loss, device = self.num_quantizers, self.quantize_dropout_multiple_of, exists(
            indices), x.device

        x = self.project_in(x)

        assert not (self.accept_image_fmap and exists(indices))

        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []
        all_quantized_out = []

        if isinstance(indices, list):
            indices = torch.stack(indices)

        if return_loss:
            assert not torch.any(
                indices == -1), 'some of the residual vq indices were dropped out. please use indices derived when the module is in eval mode to derive cross entropy loss'
            ce_losses = []

        should_quantize_dropout = self.training and self.quantize_dropout and not return_loss

        # sample a layer index at which to dropout further residual quantization
        # also prepare null indices and loss

        if should_quantize_dropout:

            # check if seed is manually passed in

            if not exists(rand_quantize_dropout_fixed_seed):
                rand_quantize_dropout_fixed_seed = get_maybe_sync_seed(device)

            rand = random.Random(rand_quantize_dropout_fixed_seed)

            rand_quantize_dropout_index = rand.randrange(self.quantize_dropout_cutoff_index, num_quant)

            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = round_up_multiple(rand_quantize_dropout_index + 1,
                                                                quant_dropout_multiple_of) - 1

            null_indices_shape = (x.shape[0], *x.shape[-2:]) if self.accept_image_fmap else tuple(x.shape[:2])
            null_indices = torch.full(null_indices_shape, -1., device=device, dtype=torch.long)
            null_loss = torch.full((1,), 0., device=device, dtype=x.dtype)

        # setup the mlps for implicit neural codebook

        maybe_code_transforms = (None,) * len(self.layers)

        if self.implicit_neural_codebook:
            maybe_code_transforms = (None, *self.mlps)

        # save all inputs across layers, for use during expiration at end under shared codebook setting

        all_residuals = []

        # go through the layers

        for quantizer_index, (vq, maybe_mlp) in enumerate(zip(self.layers, maybe_code_transforms)):

            if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                all_indices.append(null_indices)
                all_losses.append(null_loss)
                continue

            layer_indices = None
            if return_loss:
                layer_indices = indices[..., quantizer_index]

            # setup the transform code function to be passed into VectorQuantize forward

            if exists(maybe_mlp):
                maybe_mlp = partial(maybe_mlp, condition=quantized_out)

            # save for expiration

            all_residuals.append(residual)

            # vector quantize forward

            quantized, *rest = vq(
                residual,
                mask=mask,
                indices=layer_indices,
                sample_codebook_temp=sample_codebook_temp,
                freeze_codebook=freeze_codebook,
                codebook_transform_fn=maybe_mlp
            )

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            if return_quantized_out:
                all_quantized_out.append(quantized)

            if return_loss:
                ce_loss = rest[0]
                ce_losses.append(ce_loss)
                continue

            embed_indices, loss = rest

            all_indices.append(embed_indices)
            all_losses.append(loss)

        # if shared codebook, update ema only at end

        if self.training and self.shared_codebook:
            shared_layer = first(self.layers)
            shared_layer._codebook.update_ema()
            shared_layer.update_in_place_optimizer()
            shared_layer.expire_codes_(torch.cat(all_residuals, dim=-2))

        # project out, if needed

        quantized_out = self.project_out(quantized_out)

        # whether to early return the cross entropy loss

        if return_loss:
            return quantized_out, sum(ce_losses)

        # stack all losses and indices

        all_losses, all_indices = map(partial(torch.stack, dim=-1), (all_losses, all_indices))

        ret = (quantized_out, all_indices, all_losses)

        if return_all_codes:
            # whether to return all codes from all codebooks across layers
            all_codes = self.get_codes_from_indices(all_indices)

            # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)
            ret = (*ret, all_codes)

        if return_quantized_out:
            all_quantized_out = torch.stack(all_quantized_out, dim=2)
            all_quantized_out = self.project_out(all_quantized_out)
            ret = (*ret, all_quantized_out)

        return ret
