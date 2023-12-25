from predict import MolPredict
import pandas as pd
clf = MolPredict(load_model='./dataset/Scaffold/FDS')
test_pred = clf.predict('./dataset/Scaffold/test.csv')
test_results = pd.DataFrame({'pred':test_pred.flatten(),
                           'smiles':clf.datahub.data['smiles']
                            })
print(test_results.head())
test_results.to_csv("./Scaffold_FDS.csv")