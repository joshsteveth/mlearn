package ml

import (
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func newLogisticReg(alpha float64) *LReg {
	x, _ := NewMatrix([][]float64{
		[]float64{1, 2, 3},
		[]float64{2, 3, 4},
		[]float64{3, 3, 3},
	})

	y := NewVector([]float64{1, 0, 0})
	theta := NewVector([]float64{0, 0, 0})

	result, _ := NewLogisticRegression(x, y, theta, alpha)
	return result
}

func TestValidateNewLogisticRegression(t *testing.T) {
	xinput1 := [][]float64{
		[]float64{1, 2, 3},
		[]float64{3, 2, 1},
	}

	x1, _ := NewMatrix(xinput1)

	xinput2 := [][]float64{
		[]float64{1, 2, 3},
		[]float64{3, 2, 1},
		[]float64{3, 3, 3},
	}

	x2, _ := NewMatrix(xinput2)

	y1 := NewVector([]float64{1, 2, 3})
	y2 := NewVector([]float64{1, 0, 0})

	theta1 := NewVector([]float64{1, 2, 3, 4})
	theta2 := NewVector([]float64{1, 2, 3, 4, 5})

	alpha1 := float64(0)
	alpha2 := float64(1)

	//1.validation
	_, err := NewLogisticRegression(&Matrix{}, y1, theta1, alpha2)
	assert.Error(t, err)

	//2.validation
	_, err = NewLogisticRegression(x1, y2, theta1, alpha2)
	assert.Error(t, err)

	x2, _ = NewMatrix(xinput2)

	//3.validation
	_, err = NewLogisticRegression(x2, y2, theta2, alpha2)
	assert.Error(t, err)

	x2, _ = NewMatrix(xinput2)

	//4.validation
	_, err = NewLogisticRegression(x2, y2, theta1, alpha1)
	assert.Error(t, err)

	x2, _ = NewMatrix(xinput2)

	//5.validation
	_, err = NewLogisticRegression(x2, y1, theta1, alpha2)
	assert.Error(t, err)

	x2, _ = NewMatrix(xinput2)

	//all correct
	_, err = NewLogisticRegression(x2, y2, theta1, alpha2)
	assert.NoError(t, err)

}

func TestLogisticRegressionCalculateResult(t *testing.T) {
	t.Skip()
	lreg := newLogisticReg(1.5)
	fmt.Println(lreg)

	x := NewVector([]float64{3, 3, 3})
	res := lreg.h(x)
	fmt.Printf("Logistic regression result: %.5f\n", res)

	//calculate the cost function
	cost := lreg.CostFunc()
	fmt.Printf("Logistic regression cost func: %.5f\n", cost)

	fmt.Println("")
}

func TestLogisticRegressionFromExData(t *testing.T) {
	t.Skip()
	file := "data1.csv"

	//use the first 80 rows for training data
	x, _ := LoadNewMatrix(file, "1:80", "1:2")

	y, _ := LoadNewVector(file, "1:80", "3")
	theta := NewZeroVector(x.GetColumnNumber() + 1)

	lreg, err := NewLogisticRegression(x, y, theta, 0.001)
	assert.NoError(t, err)

	cost := lreg.CostFunc()
	fmt.Printf("Logistic regression cost func: %.5f\n", cost)
	fmt.Println("")

	grad := lreg.CalculateGrad()
	fmt.Printf("Logistic regression grad : %s\n", grad)

	numIt := 300000
	timeNow := time.Now()
	lreg.UpdateGrad(numIt, true)
	fmt.Printf("Time needed for %d iterations: %.0fs\n", numIt, time.Since(timeNow).Seconds())
	fmt.Printf("New logistic regression theta: %s\n", lreg.theta)
	fmt.Printf("New logistic regression cost func: %.5f\n", lreg.CostFunc())

	//use the remaining 20 for verification
	xverif, _ := LoadNewMatrix(file, "81:100", "1:2")
	yverif, _ := LoadNewVector(file, "81:100", "3")
	thetaverif := lreg.theta
	lregverif, _ := NewLogisticRegression(xverif, yverif, thetaverif, 1)

	var predictTrue, predictFalse int
	for i := 1; i <= xverif.GetRowNumber(); i++ {
		xvec, _ := xverif.GetRowVector(i)
		res := lregverif.h(xvec)

		var pred float64
		if res > 0.5 {
			pred = 1
		}
		fmt.Printf("Result: %.2f | y = %.0f | Prediction: %.0f\n",
			res, yverif.getSingleValue(i), pred)

		if pred == yverif.getSingleValue(i) {
			predictTrue += 1
		} else {
			predictFalse += 1
		}
	}

	fmt.Printf("Total true: %d; Total false: %d; precision: %.2f\n",
		predictTrue, predictFalse, float64(predictTrue)/float64(predictTrue+predictFalse))

	fmt.Println("")
}

func TestLogisticRegressionWithRegularization(t *testing.T) {
	t.Skip()
	file := "data1.csv"

	//use the first 80 rows for training data
	x, _ := LoadNewMatrix(file, "1:80", "1:2")

	y, _ := LoadNewVector(file, "1:80", "3")
	theta := NewZeroVector(x.GetColumnNumber() + 1)

	lreg, err := NewLogisticRegression(x, y, theta, 0.001)
	assert.NoError(t, err)

	lambda := float64(1)
	lreg.AddRegularizationFactor(lambda)

	cost := lreg.CostFunc()
	fmt.Printf("Logistic regression with regularization cost func: %.5f\n", cost)
	fmt.Println("")

	grad := lreg.CalculateGrad()
	fmt.Printf("Logistic regression with regularization grad : %s\n", grad)

	numIt := 25000
	timeNow := time.Now()
	lreg.UpdateGrad(numIt, true)
	fmt.Printf("Time needed for %d iterations: %.0fs\n", numIt, time.Since(timeNow).Seconds())
	fmt.Printf("New logistic regression with regularization theta: %s\n", lreg.theta)
	fmt.Printf("New logistic regression with regularization cost func: %.5f\n", lreg.CostFunc())

	//use the remaining 20 for verification
	xverif, _ := LoadNewMatrix(file, "81:100", "1:2")
	yverif, _ := LoadNewVector(file, "81:100", "3")
	thetaverif := lreg.theta
	lregverif, _ := NewLogisticRegression(xverif, yverif, thetaverif, 1)

	var predictTrue, predictFalse int
	for i := 1; i <= xverif.GetRowNumber(); i++ {
		xvec, _ := xverif.GetRowVector(i)
		res := lregverif.h(xvec)

		var pred float64
		if res > 0.5 {
			pred = 1
		}
		fmt.Printf("Result: %.2f | y = %.0f | Prediction: %.0f | Correct?: %t\n",
			res, yverif.getSingleValue(i), pred, pred == yverif.getSingleValue(i))

		if pred == yverif.getSingleValue(i) {
			predictTrue += 1
		} else {
			predictFalse += 1
		}
	}

	fmt.Printf("Total true: %d; Total false: %d; precision with regularization: %.2f\n",
		predictTrue, predictFalse, float64(predictTrue)/float64(predictTrue+predictFalse))

	//use function calculate result
	x2, _ := LoadNewMatrix(file, "81:100", "1:2")
	x2.AddConstantVectorToFirst(1)
	y2, err := lreg.CalculateResult(x2)
	assert.NoError(t, err)
	fmt.Printf("Prediction result: %s\n", y2)

	fmt.Println("")

}

func TestLogisticRegressionWithAddedFeatures(t *testing.T) {
	t.Skip()
	file := "data2.csv"

	numberFeature := uint(6)

	//use the first 80 rows for training data
	x1, _ := LoadNewVector(file, "1:90", "1")
	x2, _ := LoadNewVector(file, "1:90", "2")
	x, err := NewFeatureMatrix(x1, x2, numberFeature)
	assert.NoError(t, err)

	y, _ := LoadNewVector(file, "1:90", "3")
	theta := NewZeroVector(x.GetColumnNumber() + 1)

	lreg, err := NewLogisticRegression(x, y, theta, 0.001)
	assert.NoError(t, err)

	lambda := float64(0.1)
	lreg.AddRegularizationFactor(lambda)

	cost := lreg.CostFunc()
	fmt.Printf("Logistic regression with regularization cost func: %.5f\n", cost)
	fmt.Println("")

	grad := lreg.CalculateGrad()
	fmt.Printf("Logistic regression with regularization grad : %s\n", grad)

	numIt := 50000
	timeNow := time.Now()
	lreg.UpdateGrad(numIt, true)
	fmt.Printf("Time needed for %d iterations: %.0fs\n", numIt, time.Since(timeNow).Seconds())
	fmt.Printf("New logistic regression with regularization theta: %s\n", lreg.theta)
	fmt.Printf("New logistic regression with regularization cost func: %.5f\n", lreg.CostFunc())

	//use the remaining 20 for verification
	xverif1, _ := LoadNewVector(file, "91:118", "1")
	xverif2, _ := LoadNewVector(file, "91:118", "2")
	xverif, err := NewFeatureMatrix(xverif1, xverif2, numberFeature)
	yverif, _ := LoadNewVector(file, "91:118", "3")
	thetaverif := lreg.theta
	lregverif, _ := NewLogisticRegression(xverif, yverif, thetaverif, 1)

	var predictTrue, predictFalse int
	for i := 1; i <= xverif.GetRowNumber(); i++ {
		xvec, _ := xverif.GetRowVector(i)
		res := lregverif.h(xvec)

		var pred float64
		if res > 0.5 {
			pred = 1
		}
		fmt.Printf("Result: %.2f | y = %.0f | Prediction: %.0f | Correct?: %t\n",
			res, yverif.getSingleValue(i), pred, pred == yverif.getSingleValue(i))

		if pred == yverif.getSingleValue(i) {
			predictTrue += 1
		} else {
			predictFalse += 1
		}
	}

	fmt.Printf("Total true: %d; Total false: %d; precision with regularization: %.2f\n",
		predictTrue, predictFalse, float64(predictTrue)/float64(predictTrue+predictFalse))

	predicted, err := lreg.CalculateResult(xverif)
	assert.NoError(t, err)
	fmt.Printf("Calculated result: %s\n", predicted)

	fmt.Println("")
}
