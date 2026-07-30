"""Base-dependency geometry-viewer tests.

The complete geometry-viewer test implementation lives in
``optional_tests._test_geometry_viewer``.  This module exposes only the tests
that do not instantiate the optional pythreejs/ipympl backends.
"""

import unittest

from mcstasscript.optional_tests import _test_geometry_viewer as source


class TestPosRotFromList(source.TestPosRotFromList):
    pass


class TestNormalize(source.TestNormalize):
    pass


class TestQuaternionFromVectors(source.TestQuaternionFromVectors):
    pass


class TestQuaternionFromRotationMatrix(source.TestQuaternionFromRotationMatrix):
    pass


class TestNormalizeQuaternion(source.TestNormalizeQuaternion):
    pass


class TestQuaternionMultiply(source.TestQuaternionMultiply):
    pass


class TestTransform(source.TestTransform):
    pass


class TestShapes(source.TestShapes):
    pass


class TestTriangulateFaces(source.TestTriangulateFaces):
    pass


class TestDrawcallParsers(source.TestDrawcallParsers):
    pass


class TestComponentModel(source.TestComponentModel):
    pass


class TestInstrumentModel(source.TestInstrumentModel):
    pass


class TestConfig(source.TestConfig):
    pass


class TestApi(unittest.TestCase):
    test_get_renderer_matplotlib = source.TestApi.test_get_renderer_matplotlib
    test_get_renderer_matplotlib_3d = source.TestApi.test_get_renderer_matplotlib_3d
    test_get_renderer_matplotlib_2d = source.TestApi.test_get_renderer_matplotlib_2d
    test_get_renderer_unknown = source.TestApi.test_get_renderer_unknown
    test_matplotlib_does_not_import_pythreejs = source.TestApi.test_matplotlib_does_not_import_pythreejs
    test_json_subset_uses_dense_component_indices = source.TestApi.test_json_subset_uses_dense_component_indices
    test_json_model_builds_from_first_component_to_index_max = source.TestApi.test_json_model_builds_from_first_component_to_index_max
    test_json_rejects_reversed_component_range = source.TestApi.test_json_rejects_reversed_component_range
    test_guess_component_range_filters_rendering = source.TestApi.test_guess_component_range_filters_rendering


class TestQuaternionToRotationMatrix(source.TestQuaternionToRotationMatrix):
    pass


class TestMatplotlibTransformPoints(source.TestMatplotlibTransformPoints):
    pass


class TestMatplotlibLineSegments(source.TestMatplotlibLineSegments):
    pass


class TestIntensityToColor(source.TestIntensityToColor):
    pass


class TestAggregateIntensity(source.TestAggregateIntensity):
    pass


class TestMatplotlibIntensity(source.TestMatplotlibIntensity):
    pass


class TestMatplotlibColorbar(source.TestMatplotlibColorbar):
    pass


class TestApiComponentColors(unittest.TestCase):
    test_view_forwards_cmap_to_geometry_guess = source.TestApiComponentColors.test_view_forwards_cmap_to_geometry_guess
    test_view_guess_warning_includes_failure = source.TestApiComponentColors.test_view_guess_warning_includes_failure
    test_get_renderer_matplotlib_ignores_component_colors = source.TestApiComponentColors.test_get_renderer_matplotlib_ignores_component_colors


class TestAggregateNcount(source.TestAggregateNcount):
    pass


class TestNcountPreservesExistingBehavior(unittest.TestCase):
    test_existing_aggregations_still_work = source.TestNcountPreservesExistingBehavior.test_existing_aggregations_still_work


class TestApiComponentOpacities(unittest.TestCase):
    test_get_renderer_matplotlib_ignores_component_opacity = source.TestApiComponentOpacities.test_get_renderer_matplotlib_ignores_component_opacity


class TestGeometryRule(source.TestGeometryRule):
    pass


class TestSafeEval(source.TestSafeEval):
    pass


class TestEulerToRotationMatrix(source.TestEulerToRotationMatrix):
    pass


class TestResolveTransforms(source.TestResolveTransforms):
    pass


class TestSphereShape(unittest.TestCase):
    test_creation = source.TestSphereShape.test_creation
    test_matplotlib_render = source.TestSphereShape.test_matplotlib_render


class TestGuessGeometryBuiltins(source.TestGuessGeometryBuiltins):
    pass


class TestGeometryGuessFailureSkip(source.TestGeometryGuessFailureSkip):
    pass


class TestTransformFailureDiagnostics(source.TestTransformFailureDiagnostics):
    pass
