import pandas as pd
import numpy as np

df_test = pd.read_feather('data/input/test.feather')
df_old_test = pd.read_feather('data/input/old_test.feather')
merged_table = df_old_test.merge(df_test, on=list(df_old_test.columns.drop(['click_id'])), how='left')
print("Merged two tables: ")
print(merged_table.head())

sub_table = merged_table[~merged_table.click_id_y.isnull()]
# we only need a single old click_id for a new click_id
dictionary = {y: x for (x, y) in zip(sub_table.click_id_x, sub_table.click_id_y.astype(np.uint32))}

old_click_ids = []
new_click_ids = []
for (k, v) in dictionary.items():
    new_click_ids.append(k)
    old_click_ids.append(v)


df_mapping = pd.DataFrame({'old_click_id': old_click_ids, 'new_click_id': new_click_ids})
df_mapping.sort_values(by='old_click_id', inplace=True)
df_mapping.reset_index(drop=True).to_feather('data/working/id_mapping.feather')
