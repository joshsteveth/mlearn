package ml

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestVectorCat(t *testing.T) {
	cat := ":"

	res, err := vectorCat(cat)
	assert.NoError(t, err)
	assert.Equal(t, 0, len(res))

	cat = "a"
	_, err = vectorCat(cat)
	assert.Error(t, err)

	cat = "10"
	res, err = vectorCat(cat)
	assert.NoError(t, err)
	assert.Equal(t, 10, res[0])

	cat = "1:5"
	res, err = vectorCat(cat)
	assert.NoError(t, err)
	assert.Equal(t, 1, res[0])
	assert.Equal(t, 5, res[1])

	cat = "1:a"
	_, err = vectorCat(cat)
	assert.Error(t, err)

	cat = "1:3:5"
	_, err = vectorCat(cat)
	assert.Error(t, err)
}

func TestConvertCSVToFloat64(t *testing.T) {
	foo := [][]string{
		[]string{"34.62365962451697", "78.0246928153624", "0"},
		[]string{"30.28671076822607", "43.89499752400101", "0"},
	}

	res, err := convertCSVToFloat64(foo)
	assert.NoError(t, err)
	assert.Equal(t, 2, len(res))
	assert.Equal(t, 3, len(res[1]))
	assert.Equal(t, "30.29", fmt.Sprintf("%.2f", res[1][0]))

	bar := [][]string{
		[]string{"a", "1"},
	}

	_, err = convertCSVToFloat64(bar)
	assert.Error(t, err)
}

func TestFilterInputByCat(t *testing.T) {
	input := [][]float64{
		[]float64{1, 2, 3},
		[]float64{0, 4, 1},
	}

	res, err := filterInputByCat(input, ":", ":")
	assert.NoError(t, err)
	assert.Equal(t, 2, len(res))
	assert.Equal(t, 3, len(res[0]))

	_, err = filterInputByCat(input, ":", "4")
	assert.Error(t, err)

	res, err = filterInputByCat(input, ":", "3")
	assert.NoError(t, err)
	assert.Equal(t, 2, len(res))
	assert.Equal(t, 1, len(res[0]))
	assert.Equal(t, float64(1), res[1][0])

	_, err = filterInputByCat(input, "3", ":")
	assert.Error(t, err)

	res, err = filterInputByCat(input, "2", ":")
	assert.Equal(t, 1, len(res))
	assert.Equal(t, 3, len(res[0]))
	assert.Equal(t, float64(1), res[0][2])

	res, err = filterInputByCat(input, "1:2", "2:3")
	assert.Equal(t, 2, len(res))
	assert.Equal(t, 2, len(res[0]))
	assert.Equal(t, float64(4), res[1][0])
}

func TestSigmoidFunc(t *testing.T) {
	res := sigm(0)
	assert.Equal(t, float64(0.5), res)

	res = sigm(1)
	assert.Equal(t, "0.73", fmt.Sprintf("%.2f", res))
}
