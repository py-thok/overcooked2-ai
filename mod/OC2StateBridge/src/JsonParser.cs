using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace OC2StateBridge
{
    /// <summary>Minimal recursive-descent JSON parser for action messages.</summary>
    public static class JsonParser
    {
        private static string _s;
        private static int _i;

        public static Dictionary<string, object> ParseObject(string json)
        {
            _s = json; _i = 0;
            object v = ParseValue();
            return v as Dictionary<string, object>;
        }

        private static void SkipWs()
        {
            while (_i < _s.Length && char.IsWhiteSpace(_s[_i])) _i++;
        }

        private static object ParseValue()
        {
            SkipWs();
            if (_i >= _s.Length) return null;
            char c = _s[_i];
            if (c == '{') return ParseObj();
            if (c == '[') return ParseArr();
            if (c == '"') return ParseStr();
            if (c == 't') { Expect("true"); return true; }
            if (c == 'f') { Expect("false"); return false; }
            if (c == 'n') { Expect("null"); return null; }
            return ParseNum();
        }

        private static Dictionary<string, object> ParseObj()
        {
            Dictionary<string, object> d = new Dictionary<string, object>();
            _i++; // {
            SkipWs();
            if (_i < _s.Length && _s[_i] == '}') { _i++; return d; }
            while (_i < _s.Length)
            {
                SkipWs();
                string key = ParseStr();
                SkipWs();
                if (_i < _s.Length && _s[_i] == ':') _i++;
                d[key] = ParseValue();
                SkipWs();
                if (_i < _s.Length && _s[_i] == ',') { _i++; continue; }
                if (_i < _s.Length && _s[_i] == '}') { _i++; }
                break;
            }
            return d;
        }

        private static List<object> ParseArr()
        {
            List<object> l = new List<object>();
            _i++; // [
            SkipWs();
            if (_i < _s.Length && _s[_i] == ']') { _i++; return l; }
            while (_i < _s.Length)
            {
                l.Add(ParseValue());
                SkipWs();
                if (_i < _s.Length && _s[_i] == ',') { _i++; continue; }
                if (_i < _s.Length && _s[_i] == ']') { _i++; }
                break;
            }
            return l;
        }

        private static string ParseStr()
        {
            StringBuilder sb = new StringBuilder();
            if (_i < _s.Length && _s[_i] == '"') _i++;
            while (_i < _s.Length)
            {
                char c = _s[_i++];
                if (c == '"') break;
                if (c == '\\' && _i < _s.Length)
                {
                    char e = _s[_i++];
                    switch (e)
                    {
                        case 'n': sb.Append('\n'); break;
                        case 't': sb.Append('\t'); break;
                        case 'r': sb.Append('\r'); break;
                        case 'u':
                            if (_i + 4 <= _s.Length)
                            {
                                sb.Append((char)int.Parse(_s.Substring(_i, 4), NumberStyles.HexNumber));
                                _i += 4;
                            }
                            break;
                        default: sb.Append(e); break;
                    }
                }
                else sb.Append(c);
            }
            return sb.ToString();
        }

        private static object ParseNum()
        {
            int start = _i;
            while (_i < _s.Length && "-+0123456789.eE".IndexOf(_s[_i]) >= 0) _i++;
            double v;
            if (double.TryParse(_s.Substring(start, _i - start),
                    NumberStyles.Float, CultureInfo.InvariantCulture, out v))
                return v;
            return 0.0;
        }

        private static void Expect(string literal)
        {
            if (_i + literal.Length <= _s.Length) _i += literal.Length;
        }
    }
}
