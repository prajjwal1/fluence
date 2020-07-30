import torch
from torch import nn


class HEXProjection(nn.Module):
    def __init__(self, config):
        super(HEXProjection, self).__init__()
        self.standardize_dim = nn.Linear(config.hidden_size, config.batch_size // 2)
        self.inverse_param = nn.Parameter(torch.Tensor([0.0001]))
        self.register_buffer(
            "identity", torch.eye(config.batch_size, config.batch_size)
        )

    def forward(self, x, y):
        x = self.standardize_dim(x)  # [bs, hidden_dim] -> [hidden_dim, bs//2]
        y = self.standardize_dim(y)  # [bs, hidden_dim] -> [hidden_dim, bs//2]

        F_a = torch.cat([x, y], dim=1)  # [bs, bs]
        #  F_p = torch.cat([torch.zeros_like(x), x], dim=1)  # [bs, bs]
        F_g = torch.cat([y, torch.zeros_like(y)], dim=1)  # [bs, bs]

        internal_prod = torch.matmul(F_g.t(), F_g)
        #  assert internal_prod.shape[0] == internal_prod.shape[1]
        inverse_inside = torch.inverse(
            internal_prod + self.inverse_param * self.identity
        )

        F_l = self.identity - torch.matmul(
            torch.matmul(torch.matmul(F_g, inverse_inside), F_g.t()), F_a,
        )
        return F_l


class OrthogonalTransformer(nn.Module):
    def __init__(self, network_a, network_b, config):
        super(OrthogonalTransformer, self).__init__()
        self.network_a = network_a
        self.network_b = network_b
        self.hex = HEXProjection(config)
        self.out_1 = nn.Linear(config.batch_size, config.hidden_size)
        self.out_2 = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_tuple=None,
    ):

        output_a = self.network_a(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )[1]

        if self.training:
            output_b = self.network_b(input_ids, token_type_ids)
            projected_logits = self.hex(output_a, output_b)
            output = self.out_2(self.out_1(projected_logits))
        else:
            output = self.out_2(output_a)
        loss = self.loss_fct(output, labels)
        return loss, output
