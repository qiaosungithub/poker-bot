"""Payment cue with a fixed prior; excludes table spending summaries."""
from betting_inference_core import choose_bet_without_history as choose_bet
from betting_inference_core import choose_preflop
choose_postflop = choose_bet
