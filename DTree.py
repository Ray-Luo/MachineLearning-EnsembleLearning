import numpy as np
import pandas as pd
from collections import Counter

class DTree:
    
    def __init__(self):
        None
        
    def train(self, X, y,feature_names, forest):
        self.feature_names = feature_names
        classes = list(set(y))
        feature_names = self.feature_names
        return self.build_tree(X, y, classes, feature_names, forest=None)
    
    def build_tree(self, X, y, classes, feature_names, maxlevel=-1,level=0,forest=0):
        
        # it makes nodes and add subtrees recursively
        
        n_data = len(X)
        n_features = len(feature_names)
        
        # compute total entropy and gini df.groupby(['Activity']).size()
        class_frequency = np.zeros(len(classes))
        total_entropy = 0
        total_gini = 0
        index = 0
        
        for a_class in classes:
            if type(y) is not list:
                y = y.tolist()
            class_frequency[index] = y.count(a_class)
            total_entropy += self.cal_entropy(float(class_frequency[index]/n_data))
            total_gini += (float(class_frequency[index]/n_data))**2
            index += 1
        
        total_gini = 1 - total_gini
        default_best_feature = classes[np.argmax(class_frequency)]
        
        # base case
        if (n_data == 0 or n_features == 0 or (maxlevel>=0 and level>maxlevel)):
            return default_best_feature
        elif len(classes) == 1:
            return classes[0]
        else:
            # calculate which feature yeilds highest info gain
            info_gain = np.zeros(n_features)
            gini_gain = np.zeros(n_features)
            feature_set = range(n_features)
            if (forest != 0):
                np.random.shuffle(list(feature_set))
                feature_set = feature_set[0:forest]

            for feature in feature_set:
                entropy, gini = self.cal_info_gini_gain(X, y, classes, feature)
                info_gain[feature] = total_entropy - entropy
                gini_gain[feature] = total_gini - gini
                #print(feature_names[feature], " entropy is: ", info_gain[feature])
            
            best_feature = np.argmax(info_gain)
            tree = {feature_names[best_feature]:{}}
            #print('best_feature is:',feature_names[best_feature])

            # build a tree based on the best feature node
            # by adding a subtrees to each of possible value
            # of the best feature. When adding subtrees we need
            # to use data and features where the best feature is excluded
            values = []
            for a_x in X:
                if a_x[best_feature] not in values:
                    values.append(a_x[best_feature])

            for value in values:
                new_X = []
                new_y = y
                index = 0
                all_y = []

                for a_x in X:
                    if type(a_x) is not list:
                        a_x = a_x.tolist()
                    if a_x[best_feature] == value:
                        if best_feature == 0:
                            new_a_x = a_x[1:]
                            new_feature_names = feature_names[1:]
                        elif best_feature == n_features:
                            new_a_x = a_x[:-1]
                            new_feature_names = feature_names[:-1]
                        else:
                            new_a_x = a_x[:best_feature]
                            new_a_x.extend(a_x[best_feature+1:])
                            new_feature_names = feature_names[:best_feature]
                            new_feature_names.extend(feature_names[best_feature+1:])
                        all_y.append(y[index])
                        new_X.append(new_a_x)
                    index += 1
                    
                
                new_classes = list(set(all_y))
                subtree = self.build_tree(new_X, all_y, new_classes, new_feature_names, maxlevel, level+1, forest)
                
                tree[feature_names[best_feature]][value] = subtree

            self.tree = tree
            return tree
        
    def cal_entropy(self, prob):
        if prob != 0:
            return -prob * np.log2(prob)
        else: 
            return 0
        
    def cal_info_gini_gain(self, X, y, classes, feature):
        # calculate the info gain and gini
        n_data = len(X)
        entropy = 0
        gini_gain = 0

        unique_values = []
        for a_x in X:
            if a_x[feature] not in unique_values:
                unique_values.append(a_x[feature])
        
        # Tested
        #print('unique_values',unique_values)

        for value in unique_values:
            # count the number of feature value f with its # of different classes
            value_cnt = 0
            class_cnt = Counter()
            index = 0
            n_classes = 0
            for a_x in X:
                if a_x[feature] == value:
                    value_cnt += 1
                    n_classes += 1
                    class_cnt[y[index]] += 1
                index += 1
            
            feature_prob = float(value_cnt/n_data)

            class_entropy = 0
            gini = 0
            for ind_class in class_cnt:
                class_prob = float(class_cnt[ind_class]/n_classes)
                gini += class_prob**2
                class_entropy += self.cal_entropy(class_prob)

            entropy += feature_prob * class_entropy
            gini_gain += gini * feature_prob

        return entropy, 1 - gini_gain
    
    def predict(self, tree, data):
        if type(tree) == type("string"):
            return tree
        else:
            a = list(tree.keys())[0]
            for i in range(len(self.feature_names)):
                if self.feature_names[i] == a:
                    break
            try:
                t = tree[a][data[i]]
                return self.predict(t, data)
            except:
                return None
