setwd(normalizePath(dirname(
  R.utils::commandArgs(asValues = TRUE)$"f"
)))
source("../../scripts/h2o-r-test-setup.R")
#----------------------------------------------------------------------
# Purpose:  This Runit test aims to test the correctness of deeplearning
# mojo implementation.  A random neural network is generated and trained
# with a random dataset.  The prediction from h2o predict and mojo
# predict on a test dataset will be compared and they should equal.  Note
# that each time this test is run, it chooses the model parameters
# randomly.  We hope to test all different network settings here.
#----------------------------------------------------------------------

test.ordinalGlm.mojo <-
  function() {
    #----------------------------------------------------------------------
    # Run the test
    #----------------------------------------------------------------------
    e <- tryCatch({
      numTest = 200 # set test dataset to contain 1000 rows
      browser()
      params_prob_data <- setParmsData(numTest) # generate model parameters, random dataset
      
      modelAndDir<-buildModelSaveMojo(params_prob_data$params) # build the model and save mojo
      
      filename = sprintf("%s/in.csv", modelAndDir$dirName) # save the test dataset into a in.csv file.
      h2o.downloadCSV(params_prob_data$tDataset[,params_prob_data$params$x], filename)
      
      twoFrames<-mojoH2Opredict(modelAndDir$model, modelAndDir$dirName, filename) # perform H2O and mojo prediction and return frames
      
      compareFrames(twoFrames$h2oPredict,twoFrames$mojoPredict, prob=0.1, tolerance = 1e-4)
    }, error = function(x) x)
    if (!is.null(e))
      print("Oh, caught some error")
      print(typeof(e))
    if (!is.null(e) && (!all(sapply("DistributedException", grepl, e[[1]]))))
      FAIL(e)   # throw error unless it is unstable model error.
  }

mojoH2Opredict<-function(model, tmpdir_name, filename) {
  newTest <- h2o.importFile(filename)
  predictions1 <- h2o.predict(model, newTest)
  
  a = strsplit(tmpdir_name, '/')
  endIndex <-(which(a[[1]]=="h2o-r"))-1
  genJar <-
    paste(a[[1]][1:endIndex], collapse='/')
  
  if (.Platform$OS.type == "windows") {
    cmd <-
      sprintf(
        "java -ea -cp %s/h2o-assemblies/genmodel/build/libs/genmodel.jar -Xmx4g -XX:ReservedCodeCacheSize=256m hex.genmodel.tools.PredictCsv --header --mojo %s/%s --input %s/in.csv --output %s/out_mojo.csv",
        genJar,
        tmpdir_name,
        paste(model_key, "zip", sep = '.'),
        tmpdir_name,
        tmpdir_name
      )
  } else {
    cmd <-
      sprintf(
        "java -ea -cp %s/h2o-assemblies/genmodel/build/libs/genmodel.jar -Xmx4g -XX:ReservedCodeCacheSize=256m hex.genmodel.tools.PredictCsv --header --mojo %s/%s --input %s/in.csv --output %s/out_mojo.csv --decimal",
        genJar,
        tmpdir_name,
        paste(model@model_id, "zip", sep = '.'),
        tmpdir_name,
        tmpdir_name
      )
  }
  safeSystem(cmd)  # perform mojo prediction
  predictions2 = h2o.importFile(paste(tmpdir_name, "out_mojo.csv", sep =
                                        '/'), header=T)
  
  return(list("h2oPredict"=predictions1, "mojoPredict"=predictions2))
}

buildModelSaveMojo <- function(params) {
  model <- do.call("h2o.glm", params)
  model_key <- model@model_id
  tmpdir_name <- sprintf("%s/tmp_model_%s", sandbox(), as.character(Sys.getpid()))
  h2o.saveMojo(model, path = tmpdir_name, force = TRUE) # save mojo
  h2o.saveModel(model, path = tmpdir_name, force=TRUE) # save model to compare mojo/h2o predict offline
  
  return(list("model"=model, "dirName"=tmpdir_name))
}

setParmsData <- function(numTest=1000) {
  #----------------------------------------------------------------------
  # Parameters for the test.
  #----------------------------------------------------------------------
  missingValues <- c('MeanImputation')
  missing_values <- missingValues[sample(1:length(missingValues), replace = F)[1]]
  
  training_file <- random_dataset("multinomial", testrow = numTest)
  ratios <- (h2o.nrow(training_file)-numTest)/h2o.nrow(training_file)
  allFrames <- h2o.splitFrame(training_file, ratios)
  training_frame <- allFrames[[1]]
  test_frame <- allFrames[[2]]
  allNames = h2o.names(training_frame)
  
  params                  <- list()
  params$missing_values_handling <- missing_values
  params$training_frame <- training_frame
  params$x <- allNames[-which(allNames=="response")]
  params$y <- "response"
  params$family <- "ordinal"

  return(list("params" = params, "tDataset" = test_frame))
}

doTest("Ordinal GLM mojo test", test.ordinalGlm.mojo)
