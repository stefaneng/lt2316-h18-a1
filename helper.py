import numpy as np

def compute_prob(df_col, y):
    "Counts the values in the column and returns the probability"
    vc = df_col.value_counts()
    return vc / sum(vc)

def entropy(probs):
    return - sum(probs * np.log2(probs))

def info_gain_df(df, target_col, y_name):
    """
    Compute the information gain IG(`y_name`, `target_col`) = H(`y_name`) - H(`y_name` | `target_col`)
    This is computed using the joint defintion of H(X|Y) = - sum_{i,j} p(x_i, y_j) log [p(x_i, y_j) / p(y_j)]
    """
    #df_y = df
    #if y is not None:
    #    df_y = pd.concat([df, y], axis=1)


    n = len(df)
    # First compute the entropy for y attribute
    y_probs = df.groupby([y_name]).size().div(n)
    ent = entropy(y_probs)
    # Then compute the entropy of y, given `target_col`
    group_size = df.groupby([target_col]).size()
    joint_probs = group_size.div(n)
    cond_probs = df.groupby([target_col, y_name]).size().div(n).div(joint_probs, axis=0, level = target_col)

    return ent - sum(cond_probs.groupby(level=0).apply(entropy) * joint_probs)