## Statistics about the data (if deduplicated before/after)

### Geowac

|                               | before deduplication | after deduplication | Diff, % |
| ----------------------------- | -------------------- | ------------------- | ------- |
| sentences                     | 221,118,399          | 137,119,510         | 5.81    |
| words                         | 3,920,837,678        | 2,530,197,604       | 1.22    |
| without punctuation           | 3,272,934,051        | 2,112,278,130       | 1.13    |
| without stops                 | 3,035,021,605        | 1,953,454,263       | 1.28    |
| without stops and punctuation | 2,387,117,978        | 1,535,534,789       | 1.16    |

| Sentence diff<br> | wiki duplicates<br> | rnc duplicates<br> |
| ----------------- | ------------------- | ------------------ |
| 8,464,278         |  6,879,204          | 1,585,074          |
| %                 | 81.27%              | 18.73%             |

The table above illustrates the difference in sentences in geowac before and after deduplication with regard to the source 

### Russian wikipeia
|                               | before deduplication | after deduplication |
| ----------------------------- | -------------------- | ------------------- |
| sentences                     | 47,464,885           |  101,379,200        |
| words                         | 967,946,677          |  2,064,180,984      |
| without punctuation           | 735,695,111          |  1,569,595,654      |
| without stops                 | 838,274,197          |  1,787,156,077      |
| without stops and punctuation | 606,022,631          |  1,292,570,747      |

Total number of inner duplicated senteces in wikipedia ---  8,695,217 senteces


### RNC
|                               | before deduplication | after deduplication (inner) | duplicates rnc (inner) | after deduplication (wiki) | duplicates rnc (wiki) |
| ----------------------------- | -------------------- | --------------------------- | ---------------------- | -------------------------- | --------------------- |
| sentences                     |  18,028,159          |  18,028,159                 |  36,056,318            |  72,112,636                |  126,197,113          |
| words                         |  318,351,107         |  318,351,107                |                        |  636,702,214               |                       |
| without punctuation           |  251,508,988         |  251,508,988                |                        |  503,017,976               |                       |
| without stops                 |  231,872,956         |  231,872,956                |                        |  463,745,912               |                       |
| without stops and punctuation |  165,030,837         |  165,030,837                |                        |  330,061,674               |                       |
