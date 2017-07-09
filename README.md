# 卒業研究関連リポジトリ

## がいよー

中間発表資料や卒論，また，書いたソースコードとかを放り込んでいきます。

## TeXのふぁいるについて

### コンパイル

各提出ファイルごとに `.latexmkrc` を置いています．
環境に`LuaLaTeX`および`latexmk`が入っていれば以下でコンパイルできます．

```
$ latexmk ***.tex
```

なお， `-pvc` オプションを付けることで，変更時に自動コンパイルしてくれます（ファイル監視）．

### ゴミファイルの削除

```
$ latexmk -c
```

を実行すると，環境をきれいにしてくれます，

## こうせー

* `Docs/` 書いたものたち(卒研V報告書・中間発表・卒論 etc.)
* `Sources/` 作り上げたソースコードたち

## かんきょー

* MacBook Pro (Retina, 13-inch, Early 2015)
* CPU: Core i5-5257U (Broadwell) 2.7GHz
* RAM: 16GB
* OS: macOS Sierra(10.12.5)
* Python: 3.5.1
  * TensorFlow
  * Numpy
  * Matplotlib

