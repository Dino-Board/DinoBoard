"""pytest plugin: ark-nova-ai's board tests with the C++ engine as referee and as the source of the
board's rule answers (host_table.HostTable, HostRules).

Run from the ark-nova-ai checkout, with this directory and the DinoBoard root importable:

    PYTHONPATH=<DinoBoard>:<DinoBoard>/games/arknova/tools python -m pytest -p host_table_plugin tests/test_play_game.py
"""
import pytest

from arknova.play import rules, table
from host_table import HostRules, HostTable

# These build their position by editing game.state (pushing a frame onto it). The table's state is
# only rebuilt from the host, so the host never sees the edit.
EDIT_THE_STATE = {
    "test_conservation_claims_identify_the_chosen_tile_and_money_and_restore",
    "test_snapping_names_the_card_in_each_display_slot",
    "test_hunter_reveals_all_four_cards_but_only_animals_are_selectable",
    "test_a_sponsor_building_can_be_placed_on_the_map",
}


def pytest_configure(config):
    table.TABLE = HostTable
    rules.RULES = HostRules()


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.name in EDIT_THE_STATE:
            item.add_marker(pytest.mark.skip(reason="edits game.state, which the C++ referee does not read"))
