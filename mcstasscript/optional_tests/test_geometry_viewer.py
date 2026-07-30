"""Tests requiring the optional pythreejs and ipympl dependencies."""

import unittest

from mcstasscript.optional_tests import _test_geometry_viewer as source


class TestApi(unittest.TestCase):
    test_get_renderer_pythreejs = source.TestApi.test_get_renderer_pythreejs
    test_get_renderer_pythreejs_reports_missing_optional_module = source.TestApi.test_get_renderer_pythreejs_reports_missing_optional_module


class TestPyThreejsComponentSelection(source.TestPyThreejsComponentSelection):
    pass


class TestPyThreejsIntensity(source.TestPyThreejsIntensity):
    pass


class TestPyThreejsColorbar(source.TestPyThreejsColorbar):
    pass


class TestPyThreejsCustomColors(source.TestPyThreejsCustomColors):
    pass


class TestApiComponentColors(unittest.TestCase):
    test_get_renderer_pythreejs_passes_component_colors = source.TestApiComponentColors.test_get_renderer_pythreejs_passes_component_colors
    test_view_with_guess_renders_components = source.TestApiComponentColors.test_view_with_guess_renders_components
    test_view_with_guess_component_colors_checkbox_works = source.TestApiComponentColors.test_view_with_guess_component_colors_checkbox_works
    test_view_with_guess_no_checkbox_without_colors = source.TestApiComponentColors.test_view_with_guess_no_checkbox_without_colors
    test_view_with_guess_checkbox_present_with_colors = source.TestApiComponentColors.test_view_with_guess_checkbox_present_with_colors
    test_view_with_guess_omits_intensity_without_map = source.TestApiComponentColors.test_view_with_guess_omits_intensity_without_map
    test_view_with_guess_keeps_static_intensity_mode = source.TestApiComponentColors.test_view_with_guess_keeps_static_intensity_mode


class TestMaterialIsolation(source.TestMaterialIsolation):
    pass


class TestNcountDropdownAndLabel(source.TestNcountDropdownAndLabel):
    pass


class TestNcountPreservesExistingBehavior(unittest.TestCase):
    test_default_aggregate_is_still_total = source.TestNcountPreservesExistingBehavior.test_default_aggregate_is_still_total
    test_custom_colors_unaffected_by_ncount = source.TestNcountPreservesExistingBehavior.test_custom_colors_unaffected_by_ncount


class TestPyThreejsCustomOpacities(source.TestPyThreejsCustomOpacities):
    pass


class TestApiComponentOpacities(unittest.TestCase):
    test_get_renderer_pythreejs_passes_component_opacity = source.TestApiComponentOpacities.test_get_renderer_pythreejs_passes_component_opacity
    test_view_with_guess_component_opacity_checkbox_works = source.TestApiComponentOpacities.test_view_with_guess_component_opacity_checkbox_works
    test_view_with_guess_no_opacity_checkbox_without_opacities = source.TestApiComponentOpacities.test_view_with_guess_no_opacity_checkbox_without_opacities
    test_view_with_guess_opacity_checkbox_present_with_opacities = source.TestApiComponentOpacities.test_view_with_guess_opacity_checkbox_present_with_opacities
    test_view_with_guess_both_checkboxes_independent = source.TestApiComponentOpacities.test_view_with_guess_both_checkboxes_independent


class TestMaterialIsolationOpacity(source.TestMaterialIsolationOpacity):
    pass


class TestSphereShape(unittest.TestCase):
    test_pythreejs_render = source.TestSphereShape.test_pythreejs_render
