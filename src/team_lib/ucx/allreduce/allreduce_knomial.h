#ifndef ALLREDUCE_KNOMIAL_H_
#define ALLREDUCE_KNOMIAL_H_

enum {
    KN_BASE,
    KN_PROXY,
    KN_EXTRA
};

#define CALC_POW_K_SUP(_size, _radix, _pow_k_sup, _full_tree_size) do{  \
        int pk = 1;                                                     \
        int fs = _radix;                                                \
        while (fs < _size) {                                            \
            pk++; fs*=_radix;                                           \
        }                                                               \
        _pow_k_sup = pk;                                                \
        _full_tree_size = (fs != _size) ? fs/_radix : fs;               \
        if ((fs != _size) && (_size / _full_tree_size == 1))            \
            _pow_k_sup--;                                               \
    }while(0)

#define KN_RECURSIVE_SETUP(__radix, __myrank, __size, __pow_k_sup,      \
                           __full_tree_size, __n_full_subtrees,         \
                           __full_size, __node_type) do{                \
        CALC_POW_K_SUP(__size, __radix, __pow_k_sup, __full_tree_size); \
        __n_full_subtrees = __size / __full_tree_size;                  \
        __full_size = __n_full_subtrees*__full_tree_size;               \
        __node_type = __myrank >= __full_size ? KN_EXTRA :              \
            (__size > __full_size && __myrank < __size - __full_size ?  \
             KN_PROXY : KN_BASE);                                       \
    }while(0)

#define KN_RECURSIVE_GET_PROXY(__myrank, __full_size) (__myrank - __full_size)
#define KN_RECURSIVE_GET_EXTRA(__myrank, __full_size) (__myrank + __full_size)

enum {
    PHASE_0,
    PHASE_1,
    PHASE_EXTRA,
    PHASE_PROXY,
};

#define CHECK_PHASE(_p) case _p: goto _p; break;
#define GOTO_PHASE(_phase) do{                  \
        switch (_phase) {                       \
            CHECK_PHASE(PHASE_EXTRA);           \
            CHECK_PHASE(PHASE_PROXY);           \
            CHECK_PHASE(PHASE_1);               \
        case PHASE_0: break;                    \
        };                                      \
    } while(0)


#endif
