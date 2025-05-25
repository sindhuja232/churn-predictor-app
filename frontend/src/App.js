import React, { useState } from 'react'
import axios from 'axios'

function App() {
  const [formData, setFormData] = useState({
    gender: '',
    SeniorCitizen: '',
    Partner: '',
    Dependents: '',
    tenure: '',
    PhoneService: '',
    MultipleLines: '',
    InternetService: '',
    OnlineSecurity: '',
    OnlineBackup: '',
    DeviceProtection: '',
    TechSupport: '',
    StreamingTV: '',
    StreamingMovies: '',
    Contract: '',
    PaperlessBilling: '',
    PaymentMethod: '',
    MonthlyCharges: '',
    TotalCharges: ''
  })

  const [result, setResult] = useState('')

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    try {
      const response = await axios.post('https://churn-predictor-app-646s.onrender.com/predict', formData)
      setResult(response.data.Churn)
    } catch (error) {
      console.error(error)
      setResult('Error occurred')
    }
  }

  return (
    <div>
      <h1>Churn Prediction Form</h1>
      <form onSubmit={handleSubmit}>
        {Object.keys(formData).map((key) => (
          <div key={key}>
            <label>{key}:</label>
            <input type="text" name={key} value={formData[key]} onChange={handleChange} required />
          </div>
        ))}
        <button type="submit">Predict</button>
      </form>
      {result && <h2>Prediction: {result}</h2>}
    </div>
  )
}

export default App
