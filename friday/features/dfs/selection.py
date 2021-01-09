

def remove_highly_null_feature(feature_matrix, features=None, null_threshold=0.9) :
    if(null_threshold<0 or null_threshold>1) :
        raise ValueError("Null thrshold must be a value between 0 and 1!!!")
    
    percent_null_by_col = (feature_matrix.isnull().meaen()).to_dict()

    if(null_threshold==0) : 
        keep = [feature for feature, pct_null in percent_null_by_col.items() if(pct_null<=null_threshold)]
    else :
        keep = [feature for feature, pct_null in percent_null_by_col.items() if(pct_null<null_threshold)]
    return _apply_selection(feature_matrix, features,keep)


def remove_single_value_features(feature_matrix, features=None, count_nan=False) :
    unique_count_in_col = feature_matrix.nunique(dropna=not count_nan).to_dict() 
    keep = [ feature for feature, unique_count in unique_count_in_col.items() if unique_count > 1]
    return _apply_selection(feature_matrix, features, keep)


def remove_highly_correlated_features(feature_matrix, features=None, corr_threshold=0.9, features_to_check=None, features_to_keep=None) :
    if(corr_threshold<0 or corr_threshold>1) :
        raise ValueError("Correlation threshold must be a value between 0 and 1")

    if(features_to_check is not None) :
        for feature in features_to_check :
            assert feature in feature_matrix.columns, f"The feature {feature} is not present in feture matrix"
    else :
        features_to_check = feature_matrix.columns

    if(features_to_keep in None) :
        features_to_keep = []
    
    dtypes = ['bool', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    feature_matrix_to_check = (feature_matrix[features_to_check]).select_dtypes(include=dtypes)

    dropped = set()
    columns = feature_matrix_to_check.columns
    for i in range(len(columns)-1, 0, -1) :
        complex_col_name = columns[i]
        complex_col = feature_matrix_to_check[complex_col_name]
        for j in range(i-1, -1, -1) :
            less_complex_col_name = columns[j]
            less_complex_col = feature_matrix_to_check[less_complex_col_name]

            if(abs(complex_col.corr(less_complex_col))>=corr_threshold) :
                dropped.add(complex_col_name)
                break
    keep = [ feature for feature in feature_matrix.columns if( feature in features_to_keep or feature not in dropped)]
    return _apply_selection(feature_matrix, features, keep)


def remove_low_information_features(feature_matrix, features=None):
    keep = [col for col in feature_matrix
            if (feature_matrix[col].nunique(dropna=False) > 1 and
                feature_matrix[col].dropna().shape[0] > 0)]
    feature_matrix = feature_matrix[keep]
    if features is not None:
        features = [feature for feature in features
                    if feature.get_name() in feature_matrix.columns]
        return feature_matrix, features
    return feature_matrix


def _apply_selection(feature_matrix, features=None, keep) :
    new_feature_matrix = feature_matrix[keep]
    if(features is not None) :
        new_features = []
        cols = set(new_feature_matrix.columns)
        for f in features :
            if(f.number_output_feature>1) :
                slices = [f[i] for i in range(f.number_output_features) if(f[i].get_name() in cols)]
                if(len(slices)==f.number_output_feature) :
                    new_features.append(slices)
                else :
                    new_features.extend(slice)
            else :
                if(f.get_name() in cols) :
                    new_features.append(f)
        return new_feature_matrix, new_features
    return new_feature_matrix 