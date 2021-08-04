import utils
import constants

DATASET_VERSION = 'v1'

for trait_name in constants.TRAITS:
    df, _, indices, _ = utils.read_data(DATASET_VERSION,
                                      'all',
                                      trait_name)
    data = df.loc[indices][trait_name]
    print(f'=== {trait_name.upper()} ===')
    if len(data) == 0:
        print('  EMPTY')
    else:
        print(f'  count: {len(data):.2f}')
        print(f'  mean: {data.mean():.2f}')
        print(f'  median: {data.median():.2f}')
        print(f'  std: {data.std():.2f}')
