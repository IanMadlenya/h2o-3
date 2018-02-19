package hex.glm;

import hex.DataInfo;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;
import water.DKV;
import water.MemoryManager;
import water.TestUtil;
import water.fvec.Frame;

import java.util.Random;

import static org.junit.Assert.assertEquals;

// Want to test the following:
// 1. make sure gradient calculation is correct
// 2. for Binomial, compare ordinal result with the one for binomial
// 3. test various cases of parameter settings and make sure illegal parameters are caught and dealt with
public class GLMBasicTestOrdinal extends TestUtil {
  static Frame _trainBinomialEnum;  // response is 34
  static Frame _trainBinomial;      // response is 34
  static Frame _trainMultinomialEnum; // 0, 1 enum, response is 25
  static Frame _trainMultinomial; // response 25
  static double _tol = 1e-10;   // threshold for comparison
  Random rand = new Random();



  @BeforeClass
  public static void setup() {
    stall_till_cloudsize(1);
    _trainBinomialEnum = parse_test_file("smalldata/glm_ordinal_logit/ordinal_binomial_training_set_enum_small.csv");
    convert2Enum(_trainBinomialEnum, new int[]{0,1,2,3,4,5,6,34}); // convert enum columns
    _trainBinomial = parse_test_file("smalldata/glm_ordinal_logit/ordinal_binomial_training_set_small.csv");
    convert2Enum(_trainBinomial, new int[]{34});
    _trainMultinomialEnum = parse_test_file("smalldata/glm_ordinal_logit/ordinal_multinomial_training_set_enum_small.csv");
    convert2Enum(_trainMultinomialEnum, new int[]{0,1,25});
    _trainMultinomial = parse_test_file("smalldata/glm_ordinal_logit/ordinal_multinomial_training_set_small.csv");
    convert2Enum(_trainMultinomial, new int[] {25});
  }

  public static void convert2Enum(Frame f, int[] indices) {
    for (int index=0; index < indices.length; index++) {
      f.replace(indices[index],f.vec(indices[index]).toCategoricalVec()).remove();
    }
  }

  @AfterClass
  public static void cleanUp() {
    if (_trainBinomialEnum != null)
      _trainBinomialEnum.delete();
    if (_trainBinomial != null)
      _trainBinomial.delete();
    if (_trainMultinomialEnum != null)
      _trainMultinomialEnum.delete();
    if (_trainMultinomial != null)
      _trainMultinomial.delete();
  }

  // Ordinal regression with class = 2 defaults to binomial.  Hence, they should have the same gradients at the
  // beginning of a run.
  @Test
  public void testCheckGradientBinomial() {
    checkGradientWithBinomial(_trainBinomial, 34, "C35"); // only numerical columns
    checkGradientWithBinomial(_trainBinomialEnum, 34, "C35"); // with enum and numerical columns
  }

  // test ordinal regression with few iterations to make sure our gradient calculation and update is correct
  // for ordinals with multinomial data.  Ordinal regression coefficients are compared with ones calcluated using
  // alternate calculation without the distributed framework.  The datasets contains only numerical columns.
  @Ignore
  public void testOrdinalMultinomial() {
    int nclasses = _trainMultinomial.vec(25).domain().length;  // number of response classes
    int iterNum = rand.nextInt(10)+1;   // number of iterations to test
    GLMModel.GLMParameters paramsO = new GLMModel.GLMParameters(GLMModel.GLMParameters.Family.ordinal,
            GLMModel.GLMParameters.Family.ordinal.defaultLink, new double[]{0}, new double[]{0}, 0, 0);
    paramsO._train = _trainMultinomial._key;
    paramsO._lambda = new double[]{0.01};
    paramsO._lambda_search = false;
    paramsO._response_column = "C26";
    paramsO._lambda = new double[]{1};
    paramsO._alpha = new double[]{1};
    paramsO._objective_epsilon = 1e-6;
    paramsO._beta_epsilon = 1e-4;
    paramsO._alpha = new double[]{1};
    paramsO._max_iterations = iterNum;
    paramsO._standardize = false;

    GLMModel model = new GLM(paramsO).trainModel().get();
    updateOrdinalCoeff( _trainMultinomial, 25, paramsO);

  }

  public void updateOrdinalCoeff(Frame fr, int respCol, GLMModel.GLMParameters params) {

  }


  public void checkGradientWithBinomial(Frame fr, int respCol, String resp) {
    DataInfo dinfo=null;
    DataInfo odinfo = null;
    try {
      int nclasses = fr.vec(respCol).domain().length;
      GLMModel.GLMParameters params = new GLMModel.GLMParameters(GLMModel.GLMParameters.Family.binomial,
              GLMModel.GLMParameters.Family.binomial.defaultLink, new double[]{0}, new double[]{0}, 0, 0);
      // params._response = fr.find(params._response_column);
      params._train = fr._key;
      params._lambda = new double[]{1};
      params._lambda_search = false;
      params._response_column = resp;
      dinfo = new DataInfo(fr, null, 1,
              params._use_all_factor_levels || params._lambda_search,
              params._standardize ? DataInfo.TransformType.STANDARDIZE : DataInfo.TransformType.NONE,
              DataInfo.TransformType.NONE, true, false, false,
              false, false, false);
      DKV.put(dinfo._key, dinfo);
      GLMModel.GLMParameters paramsO = new GLMModel.GLMParameters(GLMModel.GLMParameters.Family.ordinal,
              GLMModel.GLMParameters.Family.ordinal.defaultLink, new double[]{0}, new double[]{0}, 0, 0);
      paramsO._train = fr._key;
      paramsO._lambda = new double[]{0.01};
      paramsO._lambda_search = false;
      paramsO._response_column = resp;
      paramsO._lambda = new double[]{1};
      paramsO._alpha = new double[]{1};
      odinfo = new DataInfo(fr, null, 1,
              paramsO._use_all_factor_levels || paramsO._lambda_search,
              paramsO._standardize ? DataInfo.TransformType.STANDARDIZE : DataInfo.TransformType.NONE,
              DataInfo.TransformType.NONE, true, false, false,
              false, false, false);
      DKV.put(odinfo._key, odinfo);
      double[][] _betaMultinomial = new double[nclasses][];
      for (int i = 0; i < nclasses; ++i)
        _betaMultinomial[i] = MemoryManager.malloc8d(odinfo.fullN() + 1);
      double[] beta = new double[_betaMultinomial[0].length];

      GLMTask.GLMGradientTask grBinomial = new GLMTask.GLMBinomialGradientTask(null, dinfo, params,
              1.0, beta).doAll(dinfo._adaptedFrame);
      GLMTask.GLMMultinomialGradientTask grOrdinal = new GLMTask.GLMMultinomialGradientTask(null, odinfo, 1.0,
              _betaMultinomial, 1.0, paramsO._link, paramsO).doAll(odinfo._adaptedFrame);
      compareBinomalOrdinalGradients(grBinomial, grOrdinal);  // compare and make sure the two gradients agree

    } finally {
      dinfo.remove();
      odinfo.remove();
    }
  }

  public void compareBinomalOrdinalGradients(GLMTask.GLMGradientTask bGr, GLMTask.GLMMultinomialGradientTask oGr) {
    // compare likelihood
    assertEquals(bGr._likelihood, oGr._likelihood, _tol);

    // compare gradients
    double[] binomialG = bGr._gradient;
    double[] ordinalG = oGr.gradient();

    for (int index = 0; index < binomialG.length; index++) {
      assertEquals(binomialG[index], ordinalG[index], _tol);
    }
  }
}
