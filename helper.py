import numpy as np

def compute_prob(df_col, y):
    "Counts the values in the column and returns the probability"
    vc = df_col.value_counts()
    return vc / sum(vc)

def entropy(probs):
    return - sum(probs * np.log2(probs))

def binary_split_cont(X, attr, target_attr):
    """Takes a data frame `X`, with a continuous attribute `attr`
    Creates a binary split in the data by calculating the information gain
    when splitting the data between each point
    """
    sorted_df = X.sort_values(by=[attr])
    max_gain = -np.inf
    max_midpoint = None
    max_col = None
    for i in range(len(sorted_df[attr]) - 1):
        # Compute the midpoint between each of the sorted values
        midpoint = (sorted_df[attr].iloc[i + 1] + sorted_df[attr].iloc[i]) / 2
        col_name = attr + '_' + str(i)
        # Split the data into <= midpoint and > midpoint
        sorted_df[col_name] = sorted_df[attr] <= midpoint
        # Compute the information gain with this split
        ig = info_gain_df(sorted_df, col_name, target_attr)
        # Calculate the max characteristics
        if ig > max_gain:
            max_gain = ig
            max_midpoint = midpoint
            max_col = col_name

    return (max_gain, max_midpoint, sorted_df[max_col])

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
