using System.Text;

namespace OC2StateBridge
{
    /// <summary>Minimal JSON writer (net35 has no built-in JSON).</summary>
    public class JsonWriter
    {
        private readonly StringBuilder _sb;
        private bool _needsComma;

        public JsonWriter(StringBuilder sb) { _sb = sb; }

        private void Comma()
        {
            if (_needsComma) _sb.Append(',');
            _needsComma = true;
        }

        public void BeginObject() { Comma(); _sb.Append('{'); _needsComma = false; }
        public void EndObject() { _sb.Append('}'); _needsComma = true; }
        public void BeginArray() { Comma(); _sb.Append('['); _needsComma = false; }
        public void EndArray() { _sb.Append(']'); _needsComma = true; }

        public void Key(string k)
        {
            Comma();
            _sb.Append('"').Append(k).Append("\":");
            _needsComma = false;
        }

        public void Value(string s)
        {
            Comma();
            if (s == null) { _sb.Append("null"); return; }
            _sb.Append('"');
            foreach (char c in s)
            {
                switch (c)
                {
                    case '"': _sb.Append("\\\""); break;
                    case '\\': _sb.Append("\\\\"); break;
                    case '\n': _sb.Append("\\n"); break;
                    case '\r': _sb.Append("\\r"); break;
                    case '\t': _sb.Append("\\t"); break;
                    default:
                        if (c < ' ') _sb.Append("\\u").Append(((int)c).ToString("x4"));
                        else _sb.Append(c);
                        break;
                }
            }
            _sb.Append('"');
        }

        public void Value(int v) { Comma(); _sb.Append(v); }
        public void Value(long v) { Comma(); _sb.Append(v); }
        public void Value(bool v) { Comma(); _sb.Append(v ? "true" : "false"); }
        public void Value(float v)
        {
            Comma();
            // R format round-trips; guard against NaN/Infinity which are invalid JSON
            if (float.IsNaN(v) || float.IsInfinity(v)) _sb.Append("null");
            else _sb.Append(v.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
        }
        public void Null() { Comma(); _sb.Append("null"); }
    }
}
