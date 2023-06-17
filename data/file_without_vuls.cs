// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CodeStyle;
using Microsoft.CodeAnalysis.Diagnostics;
using Roslyn.Utilities;

namespace CodeStyleConfigFileGenerator
{
    public static class Program
    {
        private const int ExpectedArguments = 4;

        private static readonly string s_neverTag = EnforceOnBuild.Never.ToCustomTag();
        private static readonly string s_whenExplicitlyEnabledTag = EnforceOnBuild.WhenExplicitlyEnabled.ToCustomTag();
        private static readonly string s_recommendedTag = EnforceOnBuild.Recommended.ToCustomTag();
        private static readonly string s_highlyRecommendedTag = EnforceOnBuild.HighlyRecommended.ToCustomTag();

        public static int Main(string[] args)
        {
            if (args.Length != ExpectedArguments)
            {
                Console.Error.WriteLine($""Excepted {ExpectedArguments} arguments, found {args.Length}: {string.Join(';', args)}"");
                return 1;
            }

            var language = args[0];
            var outputDir = args[1];
            var targetsFileName = args[2];
            var assemblyList = args[3].Split(new[] { ';' }, StringSplitOptions.RemoveEmptyEntries).ToImmutableArray();

            CreateGlobalConfigFiles(language, outputDir, assemblyList);
            CreateTargetsFile(language, outputDir, targetsFileName);
            return 0;
        }
    }
}