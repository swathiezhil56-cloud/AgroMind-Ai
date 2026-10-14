import { useState } from "react";
import { Link } from "react-router-dom";
import "./Irrigation.css";

function Irrigation() {
  const [soilMoisture, setSoilMoisture] = useState("");
  const [crop, setCrop] = useState("Rice");
  const [temperature, setTemperature] = useState("");
  const [area, setArea] = useState("");
  const [result, setResult] = useState(null);

  const handleSubmit = (e) => {
    e.preventDefault();

    if (!soilMoisture || !temperature || !area) {
      alert("Please fill all the details.");
      return;
    }

    const moisture = Number(soilMoisture);
    const temp = Number(temperature);
    const farmArea = Number(area);

    let irrigationStatus = "";
    let waterAmount = "";

    if (moisture < 30) {
      irrigationStatus = "High Water Requirement";
      waterAmount = `${Math.round(farmArea * 4500)} litres`;
    } else if (moisture < 60) {
      irrigationStatus = "Moderate Water Requirement";
      waterAmount = `${Math.round(farmArea * 3000)} litres`;
    } else {
      irrigationStatus = "Low Water Requirement";
      waterAmount = `${Math.round(farmArea * 1500)} litres`;
    }

    let advice = "";

    if (moisture < 30) {
      advice =
        "Soil moisture is low. Irrigation is recommended soon to maintain healthy crop growth.";
    } else if (moisture < 60) {
      advice =
        "Soil moisture is moderate. Schedule irrigation based on crop requirements.";
    } else {
      advice =
        "Soil moisture is sufficient. Avoid unnecessary irrigation to conserve water.";
    }

    if (temp > 35) {
      advice += " High temperature detected, so monitor the field more frequently.";
    }

    setResult({
      crop,
      irrigationStatus,
      waterAmount,
      advice,
      temperature: temp,
      moisture,
    });
  };

  return (
    <div className="irrigation-page">

      <header className="irrigation-topbar">
        <Link to="/dashboard" className="back-link">
          ← Dashboard
        </Link>

        <div className="irrigation-brand">
          🌱 AgroMind <span>AI</span>
        </div>
      </header>

      <main className="irrigation-container">

        <div className="irrigation-heading">
          <div className="irrigation-icon">💧</div>

          <h1>Smart Irrigation</h1>

          <p>
            Get intelligent irrigation suggestions based on your
            soil and farm conditions.
          </p>
        </div>

        <form
          className="irrigation-card"
          onSubmit={handleSubmit}
        >

          <div className="form-group">
            <label>🌾 Select Crop</label>

            <select
              value={crop}
              onChange={(e) => setCrop(e.target.value)}
            >
              <option value="Rice">Rice</option>
              <option value="Wheat">Wheat</option>
              <option value="Tomato">Tomato</option>
              <option value="Potato">Potato</option>
              <option value="Chilli">Chilli</option>
              <option value="Sugarcane">Sugarcane</option>
              <option value="Cotton">Cotton</option>
            </select>
          </div>

          <div className="form-group">
            <label>💧 Soil Moisture (%)</label>

            <input
              type="number"
              min="0"
              max="100"
              placeholder="Example: 45"
              value={soilMoisture}
              onChange={(e) => setSoilMoisture(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label>🌡️ Temperature (°C)</label>

            <input
              type="number"
              placeholder="Example: 32"
              value={temperature}
              onChange={(e) => setTemperature(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label>📐 Farm Area (Acres)</label>

            <input
              type="number"
              min="0.1"
              step="0.1"
              placeholder="Example: 2"
              value={area}
              onChange={(e) => setArea(e.target.value)}
            />
          </div>

          <button type="submit">
            💧 Get Irrigation Recommendation
          </button>

          {result && (
            <div className="irrigation-result">

              <div className="result-header">
                <span>🌱</span>
                <div>
                  <h3>Irrigation Recommendation</h3>
                  <p>{result.crop} Crop</p>
                </div>
              </div>

              <div className="result-grid">

                <div className="result-box">
                  <span>💧</span>
                  <strong>{result.moisture}%</strong>
                  <small>Soil Moisture</small>
                </div>

                <div className="result-box">
                  <span>🌡️</span>
                  <strong>{result.temperature}°C</strong>
                  <small>Temperature</small>
                </div>

                <div className="result-box">
                  <span>🚿</span>
                  <strong>{result.waterAmount}</strong>
                  <small>Estimated Water</small>
                </div>

              </div>

              <div className="recommendation-status">
                <strong>{result.irrigationStatus}</strong>
              </div>

              <p className="recommendation-text">
                {result.advice}
              </p>

            </div>
          )}

        </form>

      </main>
    </div>
  );
}

export default Irrigation;