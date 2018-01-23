package hex.genmodel.algos.glm;

import hex.genmodel.utils.ArrayUtils;

public class GlmOrdinalMojoModel extends GlmMojoModelBase {

  private int P;
  private int noff;

  GlmOrdinalMojoModel(String[] columns, String[][] domains, String responseColumn) {
    super(columns, domains, responseColumn);
  }

  @Override
  void init() {
    P = _beta.length / _nclasses;
    if (P * _nclasses != _beta.length)
      throw new IllegalStateException("Incorrect coding of Beta.");
    noff = _catOffsets[_cats];
  }

  @Override
  double[] glmScore0(double[] data, double[] preds) {
    preds[0] = 0;
    for (int c = 0; c < _nclasses; ++c) {
      preds[c + 1] = 0;
      if (_cats > 0) {
        if (! _useAllFactorLevels) { // skip level 0 of all factors
          for (int i = 0; i < _catOffsets.length-1; ++i) if(data[i] != 0) {
            int ival = (int) data[i] - 1;
            if (ival != data[i] - 1) throw new IllegalArgumentException("categorical value out of range");
            ival += _catOffsets[i];
            if (ival < _catOffsets[i + 1])
              preds[c + 1] += _beta[ival + c*P];
          }
        } else { // do not skip any levels
          for(int i = 0; i < _catOffsets.length-1; ++i) {
            int ival = (int) data[i];
            if (ival != data[i]) throw new IllegalArgumentException("categorical value out of range");
            ival += _catOffsets[i];
            if(ival < _catOffsets[i + 1])
              preds[c + 1] += _beta[ival + c*P];
          }
        }
      }
      for (int i = 0; i < _nums; ++i)
        preds[c+1] += _beta[noff+i + c*P]*data[i];
      preds[c+1] += _beta[(P-1) + c*P]; // reduce intercept
    }

    double expEta = Math.exp(preds[1]);
    double currProb = expEta/(1+expEta);
    double nextProb = 0;

    preds[1] = currProb;  // 0th class
    for (int c = 2; c < _nclasses; ++c) { // go class 1 to NC-2
      expEta = Math.exp(preds[c]);
      nextProb = expEta/(1+expEta);
      preds[c] = nextProb-currProb;
      currProb = nextProb;
    }
    preds[_nclasses] = 1-currProb;  // set the value to the last class
    preds[0] = 0;
    preds[0] = ArrayUtils.maxIndex(preds)-1;
    return preds;
  }
}
