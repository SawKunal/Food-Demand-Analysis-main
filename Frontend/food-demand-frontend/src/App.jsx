import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './App.css'; // Make sure this line is here

// The backend API is running on http://127.0.0.1:8000
const API_URL = 'http://127.0.0.1:8000';

function App() {
  const [formData, setFormData] = useState({
    week: 146,
    center_id: 55,
    meal_id: 1885,
    checkout_price: 158.11,
    base_price: 159.11,
    emailer: 0,
    homepage: 0,
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // --- States for Insights ---
  const [featureImportance, setFeatureImportance] = useState([]);
  const [topMeals, setTopMeals] = useState([]);
  const [categoryStats, setCategoryStats] = useState([]);

  // --- Fetch insights data on component mount ---
  useEffect(() => {
    const fetchInsights = async () => {
      try {
        const feat_res = await axios.get(`${API_URL}/insights/feature-importance`);
        setFeatureImportance(feat_res.data.feature_importance);

        const top_res = await axios.get(`${API_URL}/insights/top-historical-meals`);
        setTopMeals(top_res.data.top_meals);

        const cat_res = await axios.get(`${API_URL}/insights/category-stats`);
        setCategoryStats(cat_res.data.category_stats);

      } catch (err) {
        console.error("Error fetching insights:", err);
      }
    };
    fetchInsights();
  }, []);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? (checked ? 1 : 0) : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const response = await axios.post(`${API_URL}/predict`, {
        ...formData,
        // Ensure values are numbers
        week: Number(formData.week),
        center_id: Number(formData.center_id),
        meal_id: Number(formData.meal_id),
        checkout_price: Number(formData.checkout_price),
        base_price: Number(formData.base_price),
      });
      setResult(response.data);
    } catch (err) {
      setError("Failed to get prediction. Check console for details.");
      console.error(err);
    }
    setLoading(false);
  };

  return (
    <div className="container">
      <header>
        <h1>🍽️ Meal Demand Forecasting System</h1>
      </header>

      <main>
        <div className="form-container card">
          <h2>Get a Forecast</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-grid">
              <label>Week: <input type="number" name="week" value={formData.week} onChange={handleChange} required /></label>
              <label>Center ID: <input type="number" name="center_id" value={formData.center_id} onChange={handleChange} required /></label>
              <label>Meal ID: <input type="number" name="meal_id" value={formData.meal_id} onChange={handleChange} required /></label>
              <label>Checkout Price: <input type="number" name="checkout_price" value={formData.checkout_price} onChange={handleChange} /></label>
              <label>Base Price: <input type="number" name="base_price" value={formData.base_price} onChange={handleChange} /></label>
              <label className="checkbox">Email Promo: <input type="checkbox" name="emailer" checked={!!formData.emailer} onChange={handleChange} /></label>
              <label className="checkbox">Homepage: <input type="checkbox" name="homepage" checked={!!formData.homepage} onChange={handleChange} /></label>
            </div>
            <button type="submit" disabled={loading}>
              {loading ? 'Forecasting...' : 'Get Forecast'}
            </button>
          </form>
        </div>

        {error && <div className="card error-card"><p>{error}</p></div>}

        {result && (
          <div className="card result-card">
            <div className="result-header">
              <h3>Forecast Result</h3>
              <div className="result-kpi">
                <strong>{Math.round(result.predicted_orders)}</strong>
                <span>Predicted Orders</span>
              </div>
            </div>
            <div className="result-details">
              <p><strong>Meal:</strong> {result.meal_details.category} ({result.meal_details.cuisine})</p>
              <p><strong>Center:</strong> {result.center_id} (City: {result.center_details.city_code}, Type: {result.center_details.center_type})</p>
            </div>
            <h4>AI-Generated Explanation (from FLAN-T5):</h4>
            <pre className="explanation-box">{result.explanation}</pre>
          </div>
        )}

        <div className="insights-grid">
          <div className="card insight-card">
            <h3>Top 5 Historical Meals</h3>
            <ul>
              {topMeals.map((meal, idx) => (
                <li key={idx}>
                  <strong>{meal.category} ({meal.cuisine})</strong> at Center {meal.center_id}
                  <span>{meal.num_orders} orders (Week {meal.week})</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="card insight-card">
            <h3>Top Categories (by Avg. Demand)</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={categoryStats} layout="vertical" margin={{ left: 30 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="category" type="category" width={80} />
                <Tooltip />
                <Bar dataKey="mean" fill="#8884d8" name="Avg. Orders" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="card insight-card wide-card">
            <h3>Model Feature Importance</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={featureImportance} margin={{ bottom: 100 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="feature" angle={-45} textAnchor="end" interval={0} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="importance" fill="#82ca9d" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;