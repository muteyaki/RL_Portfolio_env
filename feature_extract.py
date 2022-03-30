from typing import Dict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
import torch.nn as nn
import torch



class Extractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict, cfg: Dict):
        super(Extractor, self).__init__(observation_space, features_dim=cfg["features_dim"])
        self.cfg = cfg
        self.name = cfg["name"]
        self._observation_space = observation_space
        self.ext_feat = nn.ModuleDict(self.module_dict())
        input_tensor = observation_space.sample()
        for item in input_tensor:
            input_tensor[item] = input_tensor[item][None]
            input_tensor[item] = torch.from_numpy(input_tensor[item])
        with torch.no_grad():
            ext_out = self.ext_forward(input_tensor)
        input_shape_linear = ext_out.shape[-1]
        self.linear = nn.Sequential(nn.Linear(input_shape_linear, cfg["features_dim"]), nn.ReLU())

    def module_dict(self) -> Dict:
        raise NotImplementedError
    
    def ext_forward(self, observations: Dict):
        raise NotImplementedError
    
    def forward(self, observations: Dict):
        ext_out = self.ext_forward(observations)
        res = self.linear(ext_out)
        return res


class LSTM(Extractor):
    """LSTM to encode time series"""
    def __init__(self, observation_space: gym.spaces.Dict, cfg: Dict):
        super(LSTM, self).__init__(observation_space, cfg)

    def ext_forward(self, observations: Dict):
        # reshape for this tensor
        data_list = []
        for item in observations:
            if "tech" in item:
                batch_len = observations[item].shape[0]
                num_layers = self.ext_feat[item].num_layers
                output_size = self.ext_feat[item].hidden_size
                h_0 = torch.randn((num_layers, batch_len, output_size), device=observations[item].device)
                c_0 = torch.randn((num_layers, batch_len, output_size), device=observations[item].device)
                out, _ = self.ext_feat[item](observations[item], (h_0, c_0))
                out = out.reshape((batch_len, -1))
                data_list.append(out)
            #elif "px_cov_mat" == item or "pct_cov_mat" == item:
            #    batch_len = observations[item].shape[0]
            #    out = self.ext_feat[item](observations[item])
            #    out = out.reshape((batch_len, -1))
            #    data_list.append(out)
            #elif "graph" ==item:
            #    batch_len = observations[item].shape[0]
            #    out = self.ext_feat[item](observations[item])
            #    out = out.reshape((batch_len, -1))
            #    data_list.append(out)
            else:
                pass
            #    data_list.append(observations[item])
        res = torch.cat(data_list, dim=-1)
        return res

    def module_dict(self) -> Dict:
        nn_dict = {}
        for item in self._observation_space.spaces:
            if "tech" in item:
                # time series module
                nn_dict[item] = nn.LSTM(input_size=self.cfg["feature_size"],
                                        hidden_size=self.cfg["lstm_output_size"],
                                        num_layers=self.cfg["lstm_num_layers"],
                                        batch_first=True)
            #elif "px_cov_mat" == item or "pct_cov_mat" == item:
            #    nn_dict[item] = nn.Conv2d(in_channels=1, out_channels=self.cfg["cnn_out_channels"],
            #                              kernel_size=self.cfg["cnn_kernel_size"])
            #elif "graph"==item:
            #    nn_dict[item] = nn.Conv2d(in_channels=1, out_channels=self.cfg["cnn_out_channels"],
            #                              kernel_size=self.cfg["cnn_kernel_size"])
            else:
                pass
        return nn_dict


class Transformer(Extractor):

        """
        transformer encoder to encode time series
        """

        def __init__(self, observation_space: gym.spaces.Dict, cfg: Dict):
            super(Transformer, self).__init__(observation_space, cfg)

        def module_dict(self) -> Dict:
            nn_dict = {}
            for item in self._observation_space.spaces:
                if "tech" in item:
                    transform_layer = nn.TransformerEncoderLayer(d_model=self.cfg["feature_size"],
                                                                 nhead=self.cfg["t_head"])
                    transform_encoder = nn.TransformerEncoder(transform_layer, num_layers=self.cfg["num_layers"])
                    nn_dict[item] = transform_encoder
                #elif item == "px_cov_mat" or item == "pct_cov_mat":
                #    nn_dict[item] = nn.Conv2d(in_channels=1, out_channels=self.cfg["cnn_out_channels"],
                #                              kernel_size=self.cfg["cnn_kernel_size"])
                else:
                    pass
            return nn_dict

        def ext_forward(self, observations: Dict):
            data_list = []
            for item in observations:
                if "tech" in item:
                    batch_len = observations[item].shape[0]
                    out = self.ext_feat[item](observations[item])
                    out = out.reshape((batch_len, -1))
                    data_list.append(out)
                else:
                    pass
                #    data_list.append(observations[item])
            res = torch.cat(data_list, dim=-1)
            return res


class AllCNN(Extractor):

    def __init__(self, observation_space: gym.spaces.Dict, cfg: Dict):
        super(AllCNN, self).__init__(observation_space=observation_space, cfg=cfg)

    def module_dict(self) -> Dict:
        nn_dict = {}
        tech_feature = []
        for item in self._observation_space.spaces:
            if "tech" in item:
                tech_feature.append(item)
        in_channels = len(tech_feature)
        nn_dict["tech_cnn"] = nn.Conv2d(in_channels=in_channels,
                                        out_channels=self.cfg["tech_out_channels"],
                                        kernel_size=self.cfg["tech_kernel_size"]
                                        )
        #nn_dict["pct_cov_mat"] = nn.Conv2d(in_channels=1,
        #                                  out_channels=self.cfg["cov_out_channels"],
        #                                   kernel_size=self.cfg["cov_kernel_size"]
        #                                   )
        #nn_dict["px_cov_mat"] = nn.Conv2d(in_channels=1,
        #                                  out_channels=self.cfg["cov_out_channels"],
        #                                  kernel_size=self.cfg["cov_kernel_size"]
        #                                  )
        return nn_dict

    def ext_forward(self, observations: Dict):
        input_list = []
        for item in observations:
            if "tech" in item:
                input_list.append(observations[item])
        tech_tensor = torch.stack(input_list, dim=1)
        batch_len = tech_tensor.shape[0]
        tech_output = self.ext_feat["tech_cnn"](tech_tensor)
        tech_output = tech_output.reshape(batch_len, -1)
        #pct_out = self.ext_feat["pct_cov_mat"](observations["pct_cov_mat"])
        #pct_out = pct_out.reshape(batch_len, -1)
        #px_out = self.ext_feat["px_cov_mat"](observations["px_cov_mat"])
        #px_out = px_out.reshape(batch_len, -1)
        #res = torch.cat([tech_output, pct_out, px_out,
        #                 observations["current_pct"],
        #                 observations["current_px"]], dim=-1)
        return tech_output


class CNNTransformer(Extractor):

    def __init__(self, observation_space: gym.spaces.Dict, cfg: Dict):
        super(CNNTransformer, self).__init__(observation_space=observation_space, cfg=cfg)

    def module_dict(self) -> Dict:
        nn_dict = {}
        for item in self._observation_space.spaces:
            if "tech" in item:
                transform_ecl = nn.TransformerEncoderLayer(
                    d_model=self.cfg["hidden_size"],
                    nhead=min(self.cfg["hidden_size"] // 2, 6)
                )
                nn_dict[item] = nn.Sequential(
                                        nn.LSTM(
                                            input_size=self.cfg["feature_size"],
                                            hidden_size=self.cfg["hidden_size"],
                                            num_layers=self.cfg["lstm_num_layers"],
                                            batch_first=True
                                        ),
                                        nn.TransformerEncoder(transform_ecl, num_layers=self.cfg["trans_num_layers"])
                )
            #elif item == "pct_cov_mat" or item == "px_cov_mat":
            #    nn_dict[item] = nn.Conv2d(in_channels=1, out_channels=self.cfg["cnn_out_channels"],
            #                              kernel_size=self.cfg["cnn_kernel_size"])
            else:
                pass
        return nn_dict

    def ext_forward(self, observations: Dict):
        data_list = []
        for item in observations:
            if "tech" in item:
                batch_len = observations[item].shape[0]
                output_size = self.ext_feat[item][0].hidden_size
                num_layers = self.ext_feat[item][0].num_layers
                h_0 = torch.randn((num_layers, batch_len, output_size), device=observations[item].device)
                c_0 = torch.randn((num_layers, batch_len, output_size), device=observations[item].device)
                out, _ = self.ext_feat[item][0](observations[item], (h_0, c_0))
                out = self.ext_feat[item][1](out)
                out = out.reshape((batch_len, -1))
                data_list.append(out)
            #elif "px_cov_mat" == item or "pct_cov_mat" == item:
            #    batch_len = observations[item].shape[0]
            #    out = self.ext_feat[item](observations[item])
            #    out = out.reshape((batch_len, -1))
            #f    data_list.append(out)
            else:
                pass
            #    data_list.append(observations[item])
        res = torch.cat(data_list, dim=-1)
        return res


extract_layer = {
    "LSTM": LSTM,
    "Transform": Transformer,
    "TCnn": AllCNN,
    "CnnTrans": CNNTransformer
}



