# Dataset files

The `data/` directory contains the following experiment inputs:

```text
data/
  main/
    kosarak.txt.gz
    movielens.txt.gz
    syn65k.txt.gz
    syn130k.txt.gz
    UserBehavior1300.txt.gz
  single/
    kosarak.single.n100000.txt.gz
    movielens.single.n100000.txt.gz
    syn65k.single.n100000.txt.gz
    syn130k.single.n100000.txt.gz
    UserBehavior1300.single.n100000.txt.gz
  supplement/
    syn65k_original100k_plus500k_fixed_mapping.txt.gz
    syn130k_original100k_plus500k_fixed_mapping.txt.gz
    zipf_clean_setvalued_d65536_n100000_a1p1.txt.gz
    zipf_clean_setvalued_d65536_n100000_a1p2.txt.gz
    zipf_clean_setvalued_d65536_n100000_a1p3.txt.gz
    zipf_clean_setvalued_d65536_n100000_a1p4.txt.gz
    zipf_clean_setvalued_d65536_n100000_a1p5.txt.gz
  case_studies/
    ifttt_100k.txt.gz
    ub1_party1.txt.gz
    ub1_party2.txt.gz
    ub1_party3.txt.gz
    ub_p123.txt.gz
```

| Directory | Contents |
|---|---|
| `main/` | The five set-valued datasets used in the main experiments. |
| `single/` | The single-valued inputs used in E8. |
| `supplement/` | The 600000-user synthetic inputs and the α=1.1–1.5 Zipf inputs. |
| `case_studies/` | The encoded IFTTT relations and three-party FedUB inputs. |

Each set-valued file has one user per line, with space-separated integer item IDs; each single-valued file has one item ID per line.

IFTTT contains 100000 users, and the FedUB party files contain 435637, 154213, and 10150 users, with their 600000-user union in `ub_p123.txt.gz`.

## Custom datasets

Place a UTF-8 text file at `data/main/NAME.txt` or `data/main/NAME.txt.gz`, using positive integer IDs separated by spaces:

```text
1 7 12
2 7
3 9 12 15
```

Use `--dataset NAME` and set `--users` to the desired number of users from that file.

An alternative data directory can be selected with the `BSSHHI_DATA_DIR` environment variable, preserving the same subdirectory layout.
