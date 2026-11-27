using System;
using System.Collections.Generic;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// 开放接口 HMAC-SHA256 签名工具。
///
/// 与后端约定（见《按机器人查询巡检异常-调用方速查》2.1）：
///   stringToSign = appKey \n timestamp \n nonce \n METHOD \n path \n canonicalQuery \n sha256(body)
///   sign         = hex(HMAC-SHA256(appSecret, stringToSign))
///
/// 三个最容易踩的坑，这里统一处理掉：
///   1. path 不含 nginx 的 /prod-api 前缀（后端 proxy_pass 会把它剥掉）；
///   2. canonicalQuery 按 key 升序、RFC 3986 编码（空格 %20，不是 +）；
///   3. GET 请求体为空，body hash 固定 e3b0c442...。
/// </summary>
public static class OpenApiSigner
{
    /// <summary>空请求体的 SHA-256，GET 请求固定用它。</summary>
    public const string EmptyBodySha256 =
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    /// <summary>nginx 企业端前缀，只出现在真实 URL 里，不参与签名。</summary>
    private const string GatewayPrefix = "/prod-api";

    /// <summary>
    /// 按 RFC 3986 编码：只保留 A-Z a-z 0-9 - _ . ~，其余按 UTF-8 百分号编码（大写 hex）。
    /// UnityWebRequest.EscapeURL 会把空格编成 %20 但保留规则不完全一致，
    /// 所以这里自己实现，保证和后端一致。
    /// </summary>
    public static string Rfc3986Encode(string value)
    {
        if (string.IsNullOrEmpty(value))
        {
            return string.Empty;
        }

        byte[] bytes = Encoding.UTF8.GetBytes(value);
        StringBuilder builder = new StringBuilder(bytes.Length * 3);

        foreach (byte b in bytes)
        {
            char c = (char)b;
            bool unreserved =
                (c >= 'A' && c <= 'Z') ||
                (c >= 'a' && c <= 'z') ||
                (c >= '0' && c <= '9') ||
                c == '-' || c == '_' || c == '.' || c == '~';

            if (unreserved)
            {
                builder.Append(c);
            }
            else
            {
                builder.Append('%');
                builder.Append(b.ToString("X2", CultureInfo.InvariantCulture));
            }
        }

        return builder.ToString();
    }

    /// <summary>把参数拼成签名用的 canonicalQuery（key 升序）。</summary>
    public static string BuildCanonicalQuery(SortedDictionary<string, string> parameters)
    {
        if (parameters == null || parameters.Count == 0)
        {
            return string.Empty;
        }

        StringBuilder builder = new StringBuilder();

        foreach (KeyValuePair<string, string> pair in parameters)
        {
            if (string.IsNullOrEmpty(pair.Key))
            {
                continue;
            }

            if (builder.Length > 0)
            {
                builder.Append('&');
            }

            builder.Append(Rfc3986Encode(pair.Key));
            builder.Append('=');
            builder.Append(Rfc3986Encode(pair.Value));
        }

        return builder.ToString();
    }

    /// <summary>秒级 Unix 时间戳。毫秒会被服务端直接判定过期。</summary>
    public static long GetUnixTimestampSeconds()
    {
        return (long)(DateTime.UtcNow - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)).TotalSeconds;
    }

    /// <summary>生成 16 位随机串（文档要求 16～32 位，且不可重复）。</summary>
    public static string CreateNonce()
    {
        return Guid.NewGuid().ToString("N").Substring(0, 16);
    }

    /// <summary>
    /// 计算签名。
    /// </summary>
    public static string ComputeSignature(
        string appKey,
        string appSecret,
        string timestamp,
        string nonce,
        string method,
        string path,
        string canonicalQuery,
        string bodyHash)
    {
        string stringToSign = string.Join("\n", new[]
        {
            appKey ?? string.Empty,
            timestamp ?? string.Empty,
            nonce ?? string.Empty,
            string.IsNullOrEmpty(method) ? "GET" : method,
            path ?? string.Empty,
            canonicalQuery ?? string.Empty,
            string.IsNullOrEmpty(bodyHash) ? EmptyBodySha256 : bodyHash
        });

        using (HMACSHA256 hmac = new HMACSHA256(Encoding.UTF8.GetBytes(appSecret ?? string.Empty)))
        {
            byte[] hash = hmac.ComputeHash(Encoding.UTF8.GetBytes(stringToSign));
            StringBuilder builder = new StringBuilder(hash.Length * 2);

            foreach (byte b in hash)
            {
                builder.Append(b.ToString("x2", CultureInfo.InvariantCulture));
            }

            return builder.ToString();
        }
    }

    /// <summary>
    /// 把完整接口地址拆成「协议+主机」与「参与签名的 path」。
    /// 允许调用方直接填 http://host:10002/prod-api/open/xxx，
    /// 签名时自动去掉 /prod-api，请求仍然打带前缀的真实地址。
    /// </summary>
    public static bool TrySplitEndpoint(string endpoint, out string authority, out string rawPath, out string signPath)
    {
        authority = string.Empty;
        rawPath = string.Empty;
        signPath = string.Empty;

        if (string.IsNullOrWhiteSpace(endpoint))
        {
            return false;
        }

        Uri uri;
        if (!Uri.TryCreate(endpoint.Trim(), UriKind.Absolute, out uri))
        {
            return false;
        }

        authority = uri.GetLeftPart(UriPartial.Authority);
        rawPath = string.IsNullOrEmpty(uri.AbsolutePath) ? "/" : uri.AbsolutePath;

        signPath = rawPath;
        if (signPath.StartsWith(GatewayPrefix, StringComparison.OrdinalIgnoreCase))
        {
            signPath = signPath.Substring(GatewayPrefix.Length);
        }

        if (string.IsNullOrEmpty(signPath))
        {
            signPath = "/";
        }

        return true;
    }

    /// <summary>
    /// 创建一个已签名的 GET 请求。
    /// </summary>
    /// <param name="endpoint">完整接口地址，可带 /prod-api 前缀和已有查询参数。</param>
    /// <param name="parameters">查询参数，会被并入已有参数并按 key 升序签名。</param>
    public static bool TryCreateSignedGetRequest(
        string endpoint,
        IDictionary<string, string> parameters,
        string appKey,
        string appSecret,
        out UnityWebRequest request,
        out string requestUrl,
        out string error)
    {
        request = null;
        requestUrl = string.Empty;
        error = string.Empty;

        if (string.IsNullOrWhiteSpace(appKey) || string.IsNullOrWhiteSpace(appSecret))
        {
            error = "开放接口 appKey / appSecret 未配置，无法生成签名。";
            return false;
        }

        string authority;
        string rawPath;
        string signPath;

        if (!TrySplitEndpoint(endpoint, out authority, out rawPath, out signPath))
        {
            error = "接口地址不是合法的绝对地址：" + endpoint;
            return false;
        }

        SortedDictionary<string, string> merged =
            new SortedDictionary<string, string>(StringComparer.Ordinal);

        // endpoint 里可能已经带了参数，先并入；调用方传入的参数优先覆盖。
        Uri uri;
        string query = Uri.TryCreate(endpoint.Trim(), UriKind.Absolute, out uri)
            ? uri.Query
            : string.Empty;

        if (!string.IsNullOrEmpty(query))
        {
            string trimmed = query[0] == '?' ? query.Substring(1) : query;

            foreach (string segment in trimmed.Split('&'))
            {
                if (string.IsNullOrEmpty(segment))
                {
                    continue;
                }

                int split = segment.IndexOf('=');
                string key = split < 0 ? segment : segment.Substring(0, split);
                string value = split < 0 ? string.Empty : segment.Substring(split + 1);
                merged[Uri.UnescapeDataString(key)] = Uri.UnescapeDataString(value);
            }
        }

        if (parameters != null)
        {
            foreach (KeyValuePair<string, string> pair in parameters)
            {
                if (string.IsNullOrEmpty(pair.Key))
                {
                    continue;
                }

                merged[pair.Key] = pair.Value ?? string.Empty;
            }
        }

        string canonicalQuery = BuildCanonicalQuery(merged);
        string timestamp = GetUnixTimestampSeconds().ToString(CultureInfo.InvariantCulture);
        string nonce = CreateNonce();
        string signature;

        try
        {
            signature = ComputeSignature(
                appKey.Trim(),
                appSecret.Trim(),
                timestamp,
                nonce,
                "GET",
                signPath,
                canonicalQuery,
                EmptyBodySha256);
        }
        catch (Exception exception)
        {
            error = "签名生成失败：" + exception.Message;
            return false;
        }

        requestUrl = authority + rawPath + (canonicalQuery.Length > 0 ? "?" + canonicalQuery : string.Empty);

        request = UnityWebRequest.Get(requestUrl);
        request.SetRequestHeader("Accept", "application/json");
        request.SetRequestHeader("X-App-Key", appKey.Trim());
        request.SetRequestHeader("X-Timestamp", timestamp);
        request.SetRequestHeader("X-Nonce", nonce);
        request.SetRequestHeader("X-Sign", signature);
        return true;
    }
}
