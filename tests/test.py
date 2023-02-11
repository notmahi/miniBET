import torch
import tqdm
import unittest

from behavior_transformer import BehaviorTransformer, GPT, GPTConfig

def test(conditional=True):
    obs_dim = 50
    act_dim = 8
    goal_dim = 50 if conditional else 0
    K = 32
    T = 16
    batch_size = 256

    gpt_model = GPT(
        GPTConfig(
            block_size=144,
            input_dim=obs_dim,
            n_layer=6,
            n_head=8,
            n_embd=256,
        )
    )
    cbet = BehaviorTransformer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        goal_dim=goal_dim,
        gpt_model=gpt_model,
        n_clusters=K,
        kmeans_fit_steps=5,
    )

    iterator = tqdm.trange(10)
    for i in iterator:
        obs_seq = torch.randn(batch_size, T, obs_dim)
        goal_seq = torch.randn(batch_size, T, goal_dim)
        action_seq = torch.randn(batch_size, T, act_dim)
        if 7 <= i < 9:
            eval_action, eval_loss, loss_dict = cbet(obs_seq, goal_seq, None)
        else:
            eval_action, eval_loss, loss_dict = cbet(obs_seq, goal_seq, action_seq)
            assert eval_loss is not None, "Correct loss is outputted."
            iterator.set_postfix_str(str(eval_loss.item()))
        assert eval_action.shape == (batch_size, T, act_dim), 'correct action outputted'
    return 0

class TestBeT(unittest.TestCase):
	def test_conditional_behavior_transformer(self):
		self.assertEqual(test(conditional=True), 0)

	def test_behavior_transformer(self):
		self.assertEqual(test(conditional=False), 0)
                
if __name__ == '__main__':
    unittest.main()