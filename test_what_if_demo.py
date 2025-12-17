#!/usr/bin/env python3
"""
Demo script for What-If Cost Analysis

Demonstrates the enhanced what_if_cost function with birthday and engagement signals
across county and state geographies.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.what_if import what_if_cost
from core.models.asset_signal import SignalType


async def main():
    """Run the what-if cost analysis demo."""
    print('🔮 MJ Data Scraper Suite - What-If Cost Analysis Demo')
    print('=' * 60)
    print()

    # Convert string signals to SignalType enums
    signals = [SignalType.BIRTHDAY, SignalType.ENGAGEMENT]
    geography_levels = ['county', 'state']

    print('📊 ANALYSIS PARAMETERS')
    print('-' * 30)
    print(f'Signals: {[s.value for s in signals]}')
    print(f'Geography Levels: {geography_levels}')
    print(f'Browser Pages: 0 (default)')
    print(f'Base Budget: None (unconstrained)')
    print(f'Risk Tolerance: medium (default)')
    print(f'Time Sensitivity: normal (default)')
    print(f'Quality Requirement: standard (default)')
    print()

    try:
        print('🚀 Running comprehensive cost analysis...')
        print('⏳ This may take a moment as we analyze multiple scenarios...')
        print()

        result = await what_if_cost(
            signals=signals,
            geography_levels=geography_levels
        )

        print('✅ ANALYSIS COMPLETE!')
        print('=' * 60)
        print()

        # Executive Summary
        print('📈 EXECUTIVE SUMMARY')
        print('-' * 30)
        cost_summary = result['cost_summary']
        recommendations = result['recommendations']

        print(f"💰 Cost Range: ${cost_summary['min_cost']:.2f} - ${cost_summary['max_cost']:.2f}")
        print(f"📊 Cost Variance: {cost_summary['cost_variance_coefficient']:.2f}")
        print(f"🏆 Recommended Geography: {recommendations['primary_recommendation']['geography'].upper()}")
        print(f"💸 Recommended Cost: ${recommendations['primary_recommendation']['estimated_cost']:.2f}")
        print()

        # Detailed Geography Analysis
        print('🗺️  GEOGRAPHY ANALYSIS')
        print('-' * 30)
        for geo_result in result['geography_analysis']:
            geo_name = geo_result['geography_level'].upper()
            cost = geo_result['estimated_cost']
            confidence = geo_result['cost_confidence']
            efficiency = geo_result.get('cost_efficiency_score', 0)
            scalability = geo_result.get('scalability_score', 0)
            density = geo_result.get('data_density_score', 0)
            rec_score = geo_result.get('recommendation_score', 0)

            print(f"\n📍 {geo_name}")
            print(".2f"            print(".2f"            print(".2f"            print(".2f"            print(".2f"            print(".2f"
            if geo_result.get('optimization_available'):
                savings = geo_result.get('optimization_savings', 0)
                print(".2f"
        print()

        # Strategic Recommendations
        print('🎯 STRATEGIC RECOMMENDATIONS')
        print('-' * 30)

        primary = recommendations['primary_recommendation']
        print(f'🏆 PRIMARY RECOMMENDATION: {primary["geography"].upper()}')
        print(".2f"        print(f'🎚️  Confidence Score: {primary["confidence_score"]:.2f}')
        print(f'💡 Reasoning: {primary["reasoning"]}')

        if primary.get('expected_benefits'):
            print('\n✅ Expected Benefits:')
            for benefit in primary['expected_benefits'][:3]:  # Show top 3
                print(f'   • {benefit}')

        if primary.get('potential_risks'):
            print('\n⚠️  Potential Risks:')
            for risk in primary['potential_risks'][:2]:  # Show top 2
                print(f'   • {risk}')

        # Alternative recommendations
        if recommendations.get('alternative_recommendations'):
            print(f'\n🔄 ALTERNATIVE OPTIONS:')
            for alt in recommendations['alternative_recommendations']:
                geo = alt['geography'].upper()
                cost = alt['estimated_cost']
                score = alt['confidence_score']
                print(".2f"
        print()

        # Cost Optimization Opportunities
        print('🔧 OPTIMIZATION OPPORTUNITIES')
        print('-' * 30)

        optimizations = result['optimization_opportunities']
        if optimizations:
            for i, opt in enumerate(optimizations[:3], 1):  # Show top 3
                print(f'{i}. 🎯 {opt["title"]}')
                print(f'   {opt["description"]}')
                if 'savings_amount' in opt:
                    print(".2f"                print(f'   📊 Impact: {opt["impact_level"].upper()}')
                print()
        else:
            print('✅ No significant optimization opportunities identified')
        print()

        # Risk Assessment
        print('⚠️  RISK ASSESSMENT')
        print('-' * 30)

        risk = result['risk_assessment']
        risk_level_colors = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}

        print(f'{risk_level_colors.get(risk["overall_risk_level"], "⚪")} Overall Risk Level: {risk["overall_risk_level"].upper()}')
        print(f'{risk_level_colors.get(risk["cost_uncertainty_risk"], "⚪")} Cost Uncertainty Risk: {risk["cost_uncertainty_risk"].upper()}')
        print(f'{risk_level_colors.get(risk["budget_overrun_risk"], "⚪")} Budget Overrun Risk: {risk["budget_overrun_risk"].upper()}')

        if risk.get('contingency_budget_required', 0) > 0:
            print(".2f"
        if risk.get('risk_factors'):
            print('\n⚠️  Key Risk Factors:')
            for factor in risk['risk_factors'][:3]:
                print(f'   • {factor}')
        print()

        # Comparative Insights
        print('📊 COMPARATIVE INSIGHTS')
        print('-' * 30)

        comparative = result['comparative_insights']
        cost_dist = comparative['cost_distribution']

        print(".2f"        print(".1f"        print(f"🏆 Most Cost-Effective: {cost_dist['most_cost_effective'].upper()}")
        print(f"💸 Least Cost-Effective: {cost_dist['least_cost_effective'].upper()}")

        efficiency = comparative['efficiency_comparison']
        print(f"\n⚡ Most Efficient: {efficiency['most_efficient'][0].upper()}")
        print(f"🐌 Least Efficient: {efficiency['least_efficient'][0].upper()}")

        # Trade-off highlights
        trade_offs = comparative['trade_off_analysis']
        cost_eff = trade_offs['cost_vs_efficiency']
        best_tradeoff = min(cost_eff, key=lambda x: x['trade_off_score'])
        print(".2f"        print()

        # Performance Projections
        print('🚀 PERFORMANCE PROJECTIONS')
        print('-' * 30)

        projections = result['performance_projections']
        exec_times = projections['execution_time_estimates']

        print('⏱️  Estimated Execution Times:')
        for geo, time_data in exec_times.items():
            hours = time_data['estimated_hours']
            range_min, range_max = time_data['confidence_range']
            print(".1f"                  ".1f"
        success_rates = projections['success_rate_projections']
        print('\n🎯 Projected Success Rates:')
        for geo, success_data in success_rates.items():
            rate = success_data['projected_rate']
            conf_min, conf_max = success_data['confidence_interval']
            print(".1f"                  ".1f"
        print()

        print('🎉 DEMO COMPLETE!')
        print('=' * 60)
        print('💡 The enhanced what_if_cost function provides comprehensive cost intelligence')
        print('   for strategic decision-making across different geography levels and signals.')
        print()
        print('🔗 Key Features Demonstrated:')
        print('   • Multi-geography cost analysis and comparison')
        print('   • ML-enhanced cost predictions with confidence intervals')
        print('   • Strategic recommendations with detailed reasoning')
        print('   • Cost optimization opportunities identification')
        print('   • Risk assessment and mitigation strategies')
        print('   • Performance projections and trade-off analysis')
        print('   • Implementation guidance and contingency planning')

    except Exception as e:
        print(f'❌ Error during analysis: {e}')
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
