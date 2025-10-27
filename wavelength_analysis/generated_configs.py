# Generated configurations based on baseline MMR (mmr_lambda050_variance)
# n_important_dimensions = 1
# n_bands_to_select = 3 to 180

configurations = []

# Generate 178 configurations (n_bands_to_select from 3 to 180)
for n_bands in range(3, 5):
    config = {
        'name': f'mmr_lambda050_variance_1dim_{n_bands}bands',
        'dimension_selection_method': 'pca',
        'perturbation_method': 'percentile',
        'perturbation_magnitudes': [15, 30, 45],
        'n_important_dimensions': 1,  # Fixed to 1
        'n_bands_to_select': n_bands,  # Varies from 3 to 180
        'normalization_method': 'variance',
        'use_diversity_constraint': True,
        'diversity_method': 'mmr',
        'lambda_diversity': 0.5  # Balanced: equal weight to influence and diversity
    }
    configurations.append(config)
# for n_bands in range(3, 31):
#     config = {
#         'name': f'mmr_lambda050_variance_1dim_{n_bands}bands',
#         'dimension_selection_method': 'pca',
#         'perturbation_method': 'percentile',
#         'perturbation_magnitudes': [15, 30, 45],
#         'n_important_dimensions': 1,  # Fixed to 1
#         'n_bands_to_select': n_bands,  # Varies from 3 to 180
#         'normalization_method': 'variance',
#         'use_diversity_constraint': True,
#         'diversity_method': 'mmr',
#         'lambda_diversity': 0.5  # Balanced: equal weight to influence and diversity
#     }
#     configurations.append(config)
#
# for n_bands in range(30, 180, 10):
#     config = {
#         'name': f'mmr_lambda050_variance_1dim_{n_bands}bands',
#         'dimension_selection_method': 'variance',
#         'perturbation_method': 'percentile',
#         'perturbation_magnitudes': [15, 30, 45],
#         'n_important_dimensions': 7,  # Fixed to 1
#         'n_bands_to_select': n_bands,  # Varies from 3 to 180
#         'normalization_method': 'variance',
#         'use_diversity_constraint': True,
#         'diversity_method': 'mmr',
#         'lambda_diversity': 0.8  # Balanced: equal weight to influence and diversity
#     }
#     configurations.append(config)

print(f"Generated {len(configurations)} configurations")
print(f"First configuration: n_bands_to_select = {configurations[0]['n_bands_to_select']}")
print(f"Last configuration: n_bands_to_select = {configurations[-1]['n_bands_to_select']}")
