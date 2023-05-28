# inductive-bias

・ [Debiasing the Human-Recommender System Feedback Loop in Collaborative Filtering](https://dl.acm.org/doi/abs/10.1145/3308560.3317303?casa_token=8nt5FYhQYBwAAAAA:H6nCAKv5NhVweM3EAXYgc3RWiw3Q120jJcTPZvXu_YHSpYDNn7Camt24nEXg0_G-fIY9joCM3azY)  
・ [How Algorithmic Confounding in Recommendation Systems Increases Homogeneity and Decreases Utility](https://dl.acm.org/doi/abs/10.1145/3240323.3240370)

を参考に、アルゴリズム交絡が原因で意図しないユーザーの均質化、アイテム間格差の拡大を確かめた論文実装

<img width="1508" alt="スクリーンショット 2023-05-27 18 09 13" src="https://github.com/tatsuki1107/inductive-bias/assets/79680980/0081e9f3-6b9e-42f2-9578-8588992dd277">

・ジニ係数：　アイテム間の効用格差  
・ジャッカード指数:　ユーザーの評価アイテムの類似度

## 考察とこれから

・ まず開発の目的があって、その上でこのジニ係数とジャッカード係数の変化の良し悪しを判断する必要がある。  
・ 論文では、人気度合いを傾向スコアとして重み付けしてるが論理的に普遍ではないことに注意したい。この場合、データの得られやすさはアルゴリズムが高い予測値を出すことに起因する  
・ これらの論文では、ユーザーは推薦アイテムを必ず評価する前提だったため、稼働するシステムではセレクションバイアスも考慮する必要がある。  
・ ランキングシステム構築時には、ポジションバイアスやエクスポジャーバイアスを考慮する必要があるが、そもそも使う機械学習アルゴリズムにも意図しないバイアスが潜んでることを念頭に置き研究する  
