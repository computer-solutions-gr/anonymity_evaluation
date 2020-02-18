from cox.store import Store


def initialize_cox_store(cox_dir='cox') -> Store:
    store = Store(cox_dir)
    store.add_table('experiments', {
        'k': int,
        'random_state': int,
        'Train AUC': float,
        'Validation AUC': float,
        'Test AUC': float,
        'Test MCC': float,
        'start_time': str,
        # 'runtime(sec)': float,
        'classifier': str,
        'classifier_full': str
    })
    return store
