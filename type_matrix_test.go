package ml

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewZeroMatrix(t *testing.T) {
	m := NewZeroMatrix(3, 5)
	err := m.validate()
	assert.NoError(t, err)
	assert.Equal(t, 5, m.GetColumnNumber())
	assert.Equal(t, 3, m.GetRowNumber())
}

func TestSetAndGetSingleValueMatrix(t *testing.T) {
	m := NewZeroMatrix(3, 5)
	err := m.SetSingleValue(4, 6, 10)
	assert.Error(t, err)

	err = m.SetSingleValue(2, 4, 10)
	assert.NoError(t, err)

	newVal, err := m.GetSingleValue(2, 4)
	assert.NoError(t, err)
	assert.Equal(t, float64(10), newVal)
}

func TestSetValueMatrix(t *testing.T) {
	invalidInput := [][]float64{
		[]float64{1, 2},
		[]float64{1},
	}

	_, err := NewMatrix(invalidInput)
	assert.Error(t, err)

	validInput := [][]float64{
		[]float64{1, 2, 3},
		[]float64{2, 3, 4},
	}

	m, err := NewMatrix(validInput)
	assert.NoError(t, err)

	val, err := m.GetSingleValue(2, 3)
	assert.NoError(t, err)
	assert.Equal(t, float64(4), val)
}

func TestTransposeMatrix(t *testing.T) {
	//t.Skip()
	input := [][]float64{
		[]float64{3, 4, 5},
		[]float64{1, 7, 2},
	}

	m, _ := NewMatrix(input)

	tr, err := m.Transpose()
	assert.NoError(t, err)
	assert.Equal(t, 2, tr.GetColumnNumber())
	assert.Equal(t, 3, tr.GetRowNumber())

	val, err := tr.GetSingleValue(3, 2)
	assert.NoError(t, err)
	assert.Equal(t, float64(2), val)

	fmt.Println("Original Matrix")
	fmt.Println(m)
	fmt.Println("Transposed Matrix")
	fmt.Println(tr)
}

func TestAddVariableMatrix(t *testing.T) {
	input := [][]float64{
		[]float64{3, 4, 5},
		[]float64{1, 7, 2},
	}

	m, _ := NewMatrix(input)

	m.AddVariable(3)
	assert.Equal(t, float64(10), m.getSingleValue(2, 2))
}

func TestMultiplyVariableMatrix(t *testing.T) {
	input := [][]float64{
		[]float64{3, 4, 5},
		[]float64{1, 7, 2},
	}

	m, _ := NewMatrix(input)

	m.MultiplyVariable(3)
	assert.Equal(t, float64(21), m.getSingleValue(2, 2))
}

func TestAddMatrix(t *testing.T) {
	input := [][]float64{
		[]float64{3, 4, 5},
		[]float64{1, 7, 2},
	}

	m, _ := NewMatrix(input)

	input2 := [][]float64{
		[]float64{4, 5},
		[]float64{7, 2},
	}

	m2, _ := NewMatrix(input2)

	err := m.AddMatrix(m2)
	assert.Error(t, err)

	input3 := [][]float64{
		[]float64{2, 1, 0},
		[]float64{4, -2, 3},
	}

	m3, _ := NewMatrix(input3)

	err = m.AddMatrix(m3)
	assert.NoError(t, err)

	for i := 1; i <= 3; i++ {
		for j := 1; j <= 2; j++ {
			assert.Equal(t, float64(5), m.getSingleValue(j, i))
		}
	}

	fmt.Printf("Added Matrix: (Should be 2x3 with values = 5)\n%s\n", m)
}

func TestGetRowVector(t *testing.T) {
	input := [][]float64{
		[]float64{3, 4, 5},
		[]float64{1, 7, 2},
	}

	m, _ := NewMatrix(input)

	_, err := m.GetRowVector(3)
	assert.Error(t, err)

	rowV, err := m.GetRowVector(2)
	assert.NoError(t, err)
	assert.Equal(t, float64(7), rowV.getSingleValue(2))

	fmt.Printf("Row vector should be [1,7,2]\n%s\n\n", rowV)

	allRows := m.GetAllRowVectors()
	assert.Equal(t, 2, len(allRows))
}

func TestGetColumnVectors(t *testing.T) {
	input := [][]float64{
		[]float64{3, 4, 5},
		[]float64{1, 7, 2},
	}

	m, _ := NewMatrix(input)

	_, err := m.GetColumnVector(4)
	assert.Error(t, err)

	colV, err := m.GetColumnVector(3)
	assert.NoError(t, err)
	assert.Equal(t, float64(5), colV.getSingleValue(1))

	fmt.Printf("Col vector should be [5,2]\n%s\n\n", colV)

	allColumns := m.GetAllColumnVectors()
	assert.Equal(t, 3, len(allColumns))
}

func TestMulyiplyMatrixes(t *testing.T) {
	input := [][]float64{
		[]float64{3, 4, 5},
		[]float64{1, 7, 2},
	}

	m, _ := NewMatrix(input)

	input2 := [][]float64{
		[]float64{3, 4, 5},
		[]float64{1, 7, 2},
	}

	m2, _ := NewMatrix(input2)

	_, err := m.Multiply(m2)
	assert.Error(t, err)

	input3 := [][]float64{
		[]float64{3, 4, 5},
		[]float64{1, 7, 2},
		[]float64{0, 3, 1},
	}

	m3, _ := NewMatrix(input3)

	res, err := m.Multiply(m3)
	assert.NoError(t, err)
	rowVec := res.getRowVector(2)
	assert.Equal(t, float64(59), rowVec.getSingleValue(2))

	fmt.Printf("2nd row vector of multiplication result should be [10, 59, 21]\n%s\n\n", rowVec)
}

func TestLoadCSVToMatrix(t *testing.T) {
	file := "data1.csv"
	m, err := LoadNewMatrix(file, ":", "1:2")
	assert.NoError(t, err)
	assert.Equal(t, 2, m.GetColumnNumber())
	assert.Equal(t, 100, m.GetRowNumber())
}

func TestAddRowVectorMatrix(t *testing.T) {
	input := [][]float64{
		[]float64{3, 4, 5},
		[]float64{1, 7, 2},
	}

	m, _ := NewMatrix(input)

	v := NewVector([]float64{1, 2})
	err := m.AddRowVector(v)
	assert.Error(t, err)

	v2 := NewVector([]float64{1, 2, 3})
	err = m.AddRowVector(v2)
	assert.NoError(t, err)
	assert.Equal(t, 3, m.GetRowNumber())
	assert.Equal(t, float64(3), m.getSingleValue(3, 3))
}

func TestAddColumnVectorMatrix(t *testing.T) {
	input := [][]float64{
		[]float64{3, 4, 5},
		[]float64{1, 7, 2},
	}

	m, _ := NewMatrix(input)

	v := NewVector([]float64{1, 2, 3})
	err := m.AddColumnVector(v)
	assert.Error(t, err)

	v2 := NewVector([]float64{1, 2})
	err = m.AddColumnVector(v2)
	assert.NoError(t, err)
	assert.Equal(t, 4, m.GetColumnNumber())
	assert.Equal(t, float64(1), m.getSingleValue(1, 4))
}

func TestNewFeatureMatrix(t *testing.T) {
	v1 := NewVector([]float64{1, 2, 3})
	v2 := NewVector([]float64{0, 1, 4})
	v3 := NewVector([]float64{1, 2})

	_, err := NewFeatureMatrix(v1, v3, 1)
	assert.Error(t, err)

	m, err := NewFeatureMatrix(v1, v2, 2)
	assert.NoError(t, err)
	assert.Equal(t, 6, m.GetColumnNumber())
	assert.Equal(t, 3, m.GetRowNumber())
	assert.Equal(t, float64(16), m.getSingleValue(3, 6))
	fmt.Println(m)
}

func TestAddOnesVectorToFirstMatrix(t *testing.T) {
	input := [][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 3, 2},
	}
	m, _ := NewMatrix(input)

	m.AddConstantVectorToFirst(1)
	assert.Equal(t, 2, m.GetRowNumber())
	assert.Equal(t, 4, m.GetColumnNumber())
	assert.Equal(t, float64(1), m.getSingleValue(1, 1))
	assert.Equal(t, float64(1), m.getSingleValue(2, 1))
	assert.Equal(t, float64(2), m.getSingleValue(2, 4))

}
